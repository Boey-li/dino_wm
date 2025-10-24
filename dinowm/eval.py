import os
import time
import hydra
import torch
import logging
import warnings
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from collections import OrderedDict
from torchvision import utils

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
        
        ### Load model
        self.model, self.model_cfg = self._load_model()
        self.model.eval()

        ### Load dataset
        seed(cfg.seed)
        self.dataset, self.traj_dset = hydra.utils.call(
            self.cfg.evaluation.dataset,
            num_hist=self.model_cfg.num_hist,
            num_pred=self.model_cfg.num_pred,
            frameskip=self.model_cfg.frameskip,
        )
        
    
    
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
    
    
    def run(self):
        rollout_logs = self.openloop_rollout(self.traj_dset)
        log.info("Open-loop Rollout Evaluation Results:")
        for k, v in rollout_logs.items():
            log.info(f"{k}: {v:.6f}")
    
    
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
    
    
    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
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

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{self.save_dir}/samples.png",
        )
    
    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )
    
    
################################
# Main 
################################  
@hydra.main(config_path="conf", config_name="eval")
def main(cfg: OmegaConf):    
    evaluator = ModelEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()