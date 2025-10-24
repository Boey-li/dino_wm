import os
import time
import hydra
import logging
import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from collections import OrderedDict

import torch
from torchvision import utils
from torch.utils.data import DataLoader

from metrics.image_metrics import eval_images
from utils import cfg_to_dict, seed, load_model, slice_trajdict_with_t, sample_tensors

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Evaluation results saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        
        self.save_dir = cfg_dict['saved_folder']
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.save_dir, 'eval.log')),
                logging.StreamHandler()
            ]
        )
        log.info("Starting model evaluation...")
        
        # Initialize local metrics storage
        self.eval_metrics = OrderedDict()
        
        ### Load model
        self.model, self.model_cfg = self._load_model()
        self.num_reconstruct_samples = self.model_cfg.training.num_reconstruct_samples

        ### Load dataset
        seed(cfg.seed)
        self.dataset, self.traj_dset = hydra.utils.call(
            self.cfg.evaluation.dataset,
            num_hist=self.model_cfg.num_hist,
            num_pred=self.model_cfg.num_pred,
            frameskip=self.model_cfg.frameskip,
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.model_cfg.gpu_batch_size,
            shuffle=False,
            num_workers=self.cfg.evaluation.num_workers,
            collate_fn=None,
        )
        log.info(f"dataloader batch size: {self.model_cfg.gpu_batch_size}")
        

    def _load_model(self):
        model_path = self.cfg.model_path
        with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
            model_cfg = OmegaConf.load(f)
        num_action_repeat = model_cfg.num_action_repeat
        model_ckpt = (
            Path(model_path) / 'checkpoints' / f'model_{self.cfg.model_epoch}.pth'
        )
        model = load_model(model_ckpt, model_cfg, num_action_repeat, self.device)
        return model, model_cfg
    
    
    def eval(self):
        # open loop rollout with random horizons
        self.model.eval()
        rollout_logs = self.openloop_rollout(self.traj_dset)
        log.info("Open-loop Rollout Evaluation Results:")
        for k, v in rollout_logs.items():
            log.info(f"{k}: {v:.6f}")
        
        # rollout on dataset slices with fixed horizon
        for i, data in enumerate(tqdm(self.dataloader)):
            obs, act, state = data

            for k in obs.keys():
                obs[k] = obs[k].to(self.device)
            act = act.to(self.device)
            state = state.to(self.device)
            
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )
            
            loss = self.gather_for_metrics(loss).mean()

            loss_components = self.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }
            
            # Track overall loss
            self.logs_update({'val_loss': [loss.item()]})
            
            if self.model_cfg.has_decoder:
                # only eval images when plotting due to speed
                if self.model_cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.model_cfg.num_hist, self.model_cfg.num_hist + self.model_cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.model_cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                )
                
            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)
        
        # Print final evaluation summary
        self.print_final_summary()
    
    
    def openloop_rollout(self, dset, num_rollout=10, 
                         rand_start_end=True, min_horizon=2):
        np.random.seed(self.cfg.seed)
        min_horizon = min_horizon + self.model_cfg.num_hist
        plotting_dir = os.path.join(self.save_dir, "rollout_plots")
        os.makedirs(plotting_dir, exist_ok=True)
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.model_cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.model_cfg.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.model_cfg.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.model_cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.model_cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start : 
                    start + horizon * self.model_cfg.frameskip + 1 : 
                    self.model_cfg.frameskip
                ]
            act = act[start : start + horizon * self.model_cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.model_cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                z_obses, z = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.model_cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/{idx}{postfix}.png",
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs


    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs
    
    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs
    
    
    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        batch,
        num_samples=2,
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        save_plot_dir = os.path.join(self.save_dir, "eval")
        os.makedirs(save_plot_dir, exist_ok=True)
        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{save_plot_dir}/eval_b{batch}.png",
        )
    
    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )
    
    def logs_update(self, logs):
        """Update local metrics storage"""
        for key, value in logs.items():
            if key not in self.eval_metrics:
                self.eval_metrics[key] = []
            if isinstance(value, list):
                self.eval_metrics[key].extend(value)
            else:
                self.eval_metrics[key].append(value)
    
    def gather_for_metrics(self, tensor_dict):
        """Local replacement for accelerator.gather_for_metrics"""
        if isinstance(tensor_dict, dict):
            return {k: v.detach().cpu() for k, v in tensor_dict.items()}
        else:
            return tensor_dict.detach().cpu()
    
    def print_final_summary(self):
        """Print final evaluation summary with loss and loss components"""
        log.info("=" * 60)
        log.info("FINAL EVALUATION SUMMARY")
        log.info("=" * 60)
        
        # Print overall loss
        if 'val_loss' in self.eval_metrics and self.eval_metrics['val_loss']:
            loss_values = self.eval_metrics['val_loss']
            mean_loss = np.mean(loss_values)
            std_loss = np.std(loss_values)
            log.info(f"Overall Loss: {mean_loss:.6f} ± {std_loss:.6f}")
            log.info(f"Loss Count: {len(loss_values)} batches")
        
        # Print loss components
        loss_component_keys = [k for k in self.eval_metrics.keys() if k.startswith('val_') and not k.startswith('val_img_') and not k.startswith('val_z_')]
        if loss_component_keys:
            log.info("\nLoss Components:")
            for key in loss_component_keys:
                if self.eval_metrics[key]:
                    values = self.eval_metrics[key]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    log.info(f"  {key}: {mean_val:.6f} ± {std_val:.6f}")
        
        # Print latent space metrics
        latent_keys = [k for k in self.eval_metrics.keys() if k.startswith('val_z_')]
        if latent_keys:
            log.info("\nLatent Space Metrics:")
            for key in latent_keys:
                if self.eval_metrics[key]:
                    values = self.eval_metrics[key]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    log.info(f"  {key}: {mean_val:.6f} ± {std_val:.6f}")
        
        # Print image quality metrics
        image_keys = [k for k in self.eval_metrics.keys() if k.startswith('val_img_')]
        if image_keys:
            log.info("\nImage Quality Metrics:")
            for key in image_keys:
                if self.eval_metrics[key]:
                    values = self.eval_metrics[key]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    log.info(f"  {key}: {mean_val:.6f} ± {std_val:.6f}")
        
        log.info("=" * 60)
    
    
################################
# Main 
################################  
@hydra.main(config_path="conf", config_name="eval")
def main(cfg: OmegaConf):    
    evaluator = ModelEvaluator(cfg)
    evaluator.eval()


if __name__ == "__main__":
    main()