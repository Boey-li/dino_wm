import os
import time
import json
import pickle
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

from dinowm.metrics.image_metrics import eval_images
from dinowm.utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
from dinowm.preprocessor import Preprocessor

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor", 
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

class ModelEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Evaluation results saved dir: {cfg['saved_folder']}")
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")
        
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Setup output directories
        self.setup_output_directories()
        
        # Load model
        self.model = self.load_model()
        
        # Load test dataset
        self.test_dataset, self.test_traj_dset = self.load_test_dataset()
        
        # Initialize metrics tracking
        self.evaluation_results = OrderedDict()
        
        log.info("ModelEvaluator initialized successfully")

    def setup_output_directories(self):
        """Create output directory structure"""
        self.results_dir = Path("eval_results")
        self.plots_dir = self.results_dir / "plots"
        self.logs_dir = self.results_dir / "logs"
        
        for dir_path in [self.results_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Output directories created at: {self.results_dir}")

    def load_model(self):
        """Load model from checkpoint"""
        model_path = f"{self.cfg.ckpt_base_path}/outputs/{self.cfg.model_name}/"
        
        # Load training configuration
        with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
            model_cfg = OmegaConf.load(f)
        
        # Load checkpoint
        model_ckpt = Path(model_path) / "checkpoints" / f"model_{self.cfg.model_epoch}.pth"
        
        if not model_ckpt.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt}")
        
        log.info(f"Loading model from: {model_ckpt}")
        
        # Load checkpoint data
        with model_ckpt.open("rb") as f:
            payload = torch.load(f, map_location=self.device)
        
        loaded_keys = []
        result = {}
        for k, v in payload.items():
            if k in ALL_MODEL_KEYS:
                loaded_keys.append(k)
                result[k] = v.to(self.device)
        
        result["epoch"] = payload["epoch"]
        log.info(f"Loaded model from epoch {result['epoch']}")
        log.info(f"Loaded model components: {loaded_keys}")
        
        # Instantiate missing components
        if "encoder" not in result:
            result["encoder"] = hydra.utils.instantiate(model_cfg.encoder)
        
        if "predictor" not in result:
            raise ValueError("Predictor not found in model checkpoint")
        
        if model_cfg.has_decoder and "decoder" not in result:
            if model_cfg.env.decoder_path is not None:
                decoder_path = os.path.join(self.base_path, model_cfg.env.decoder_path)
                ckpt = torch.load(decoder_path)
                if isinstance(ckpt, dict):
                    result["decoder"] = ckpt["decoder"]
                else:
                    result["decoder"] = torch.load(decoder_path)
            else:
                raise ValueError("Decoder path not found in model checkpoint and is not provided in config")
        elif not model_cfg.has_decoder:
            result["decoder"] = None
        
        # Instantiate model
        model = hydra.utils.instantiate(
            model_cfg.model,
            encoder=result["encoder"],
            proprio_encoder=result["proprio_encoder"],
            action_encoder=result["action_encoder"],
            predictor=result["predictor"],
            decoder=result["decoder"],
            proprio_dim=model_cfg.proprio_emb_dim,
            action_dim=model_cfg.action_emb_dim,
            concat_dim=model_cfg.concat_dim,
            num_action_repeat=model_cfg.num_action_repeat,
            num_proprio_repeat=model_cfg.num_proprio_repeat,
        )
        
        model.to(self.device)
        model.eval()
        
        # Store model config for later use
        self.model_cfg = model_cfg
        
        return model

    def load_test_dataset(self):
        """Load test dataset"""
        seed(self.cfg.evaluation.seed)
        
        log.info(f"Loading test dataset from {self.model_cfg.env.dataset.data_path} ...")
        
        datasets, traj_dsets = hydra.utils.call(
            self.model_cfg.env.dataset,
            num_hist=self.model_cfg.num_hist,
            num_pred=self.model_cfg.num_pred,
            frameskip=self.model_cfg.frameskip,
        )
        
        test_dataset = datasets["valid"]  # Use validation set as test set
        test_traj_dset = traj_dsets["valid"]
        
        # Create dataloader
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.evaluation.batch_size,
            shuffle=False,
            num_workers=self.cfg.evaluation.num_workers,
            collate_fn=None,
        )
        
        log.info(f"Test dataset loaded: {len(test_dataset)} samples")
        log.info(f"Test dataloader batch size: {self.cfg.evaluation.batch_size}")
        
        return test_dataloader, test_traj_dset

    def evaluate_predictions(self):
        """Main evaluation loop over test batches"""
        log.info("Starting prediction evaluation...")
        
        self.model.eval()
        batch_results = []
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_dataset, desc="Evaluating predictions")):
                obs, act, state = data
                
                # Move to device
                for k in obs.keys():
                    obs[k] = obs[k].to(self.device)
                act = act.to(self.device)
                state = state.to(self.device)
                
                # Get model predictions
                z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(obs, act)
                
                # Evaluate latent space
                if self.model_cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)
                    
                    latent_metrics = self.evaluate_latent_space(z_obs_out, z_tgt)
                else:
                    latent_metrics = {}
                
                # Evaluate image quality
                image_metrics = {}
                if visual_out is not None and self.model_cfg.has_decoder:
                    image_metrics.update(self.evaluate_image_quality(
                        visual_out, obs["visual"], prefix="prediction"
                    ))
                
                if visual_reconstructed is not None and self.model_cfg.has_decoder:
                    image_metrics.update(self.evaluate_image_quality(
                        visual_reconstructed, obs["visual"], prefix="reconstruction"
                    ))
                
                # Store batch results
                batch_result = {
                    "batch_idx": i,
                    "loss": loss.item(),
                    "loss_components": {k: v.item() for k, v in loss_components.items()},
                    "latent_metrics": latent_metrics,
                    "image_metrics": image_metrics,
                }
                
                batch_results.append(batch_result)
                
                # Generate plots for first few batches
                if i < self.cfg.evaluation.num_plot_batches and self.model_cfg.has_decoder:
                    self.plot_predictions(
                        obs["visual"], visual_out, visual_reconstructed, i
                    )
        
        # Aggregate results
        aggregated_results = self.aggregate_metrics(batch_results)
        
        log.info("Prediction evaluation completed")
        return aggregated_results

    def evaluate_latent_space(self, z_pred, z_target):
        """Evaluate latent space predictions"""
        logs = {}
        
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        
        for name, (start_idx, end_idx) in slices.items():
            z_pred_slice = slice_trajdict_with_t(z_pred, start_idx=start_idx, end_idx=end_idx)
            z_target_slice = slice_trajdict_with_t(z_target, start_idx=start_idx, end_idx=end_idx)
            
            for k in z_pred_slice.keys():
                loss = self.model.emb_criterion(z_pred_slice[k], z_target_slice[k])
                logs[f"z_{k}_err_{name}"] = loss.item()
        
        return logs

    def evaluate_image_quality(self, pred_images, target_images, prefix=""):
        """Evaluate image quality metrics"""
        metrics = {}
        
        if pred_images is None or target_images is None:
            return metrics
        
        # Evaluate prediction frames
        for t in range(self.model_cfg.num_hist, self.model_cfg.num_hist + self.model_cfg.num_pred):
            if t < pred_images.shape[1] and t < target_images.shape[1]:
                img_metrics = eval_images(
                    pred_images[:, t - self.model_cfg.num_pred], 
                    target_images[:, t]
                )
                
                for metric_name, metric_value in img_metrics.items():
                    key = f"{prefix}_img_{metric_name}_t{t}" if prefix else f"img_{metric_name}_t{t}"
                    metrics[key] = metric_value.item()
        
        # Evaluate reconstruction frames
        if prefix == "reconstruction":
            for t in range(target_images.shape[1]):
                if t < pred_images.shape[1]:
                    img_metrics = eval_images(
                        pred_images[:, t], 
                        target_images[:, t]
                    )
                    
                    for metric_name, metric_value in img_metrics.items():
                        key = f"{prefix}_img_{metric_name}_t{t}"
                        metrics[key] = metric_value.item()
        
        return metrics

    def evaluate_rollout(self):
        """Evaluate model's rollout capabilities"""
        log.info("Starting rollout evaluation...")
        
        self.model.eval()
        rollout_results = []
        
        # Sample trajectories for rollout evaluation
        num_rollouts = min(self.cfg.evaluation.num_rollouts, len(self.test_traj_dset))
        
        with torch.no_grad():
            for idx in range(num_rollouts):
                traj_idx = np.random.randint(0, len(self.test_traj_dset))
                obs, act, state, _ = self.test_traj_dset[traj_idx]
                
                # Move to device
                for k in obs.keys():
                    obs[k] = obs[k].to(self.device)
                act = act.to(self.device)
                
                # Prepare rollout data
                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = obs[k][:self.model_cfg.num_hist].unsqueeze(0)
                
                actions = act[:self.model_cfg.num_pred * self.model_cfg.frameskip].unsqueeze(0)
                actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.model_cfg.frameskip)
                
                # Perform rollout
                z_obses, _ = self.model.rollout(obs_0, actions)
                
                # Get final predictions
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                
                # Get ground truth final state
                obs_g = {}
                for k in obs.keys():
                    obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0)
                
                z_g = self.model.encode_obs(obs_g)
                
                # Compute rollout metrics
                rollout_metrics = self.evaluate_latent_space(z_obs_last, z_g)
                
                rollout_result = {
                    "traj_idx": traj_idx,
                    "rollout_metrics": rollout_metrics,
                }
                
                rollout_results.append(rollout_result)
        
        # Aggregate rollout results
        aggregated_rollout = self.aggregate_rollout_metrics(rollout_results)
        
        log.info("Rollout evaluation completed")
        return aggregated_rollout

    def aggregate_metrics(self, batch_results):
        """Aggregate metrics across batches"""
        aggregated = {
            "total_batches": len(batch_results),
            "loss": {
                "mean": np.mean([r["loss"] for r in batch_results]),
                "std": np.std([r["loss"] for r in batch_results]),
            },
            "loss_components": {},
            "latent_metrics": {},
            "image_metrics": {},
        }
        
        # Aggregate loss components
        if batch_results and "loss_components" in batch_results[0]:
            loss_comp_keys = batch_results[0]["loss_components"].keys()
            for key in loss_comp_keys:
                values = [r["loss_components"][key] for r in batch_results]
                aggregated["loss_components"][key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                }
        
        # Aggregate latent metrics
        if batch_results and "latent_metrics" in batch_results[0]:
            latent_keys = batch_results[0]["latent_metrics"].keys()
            for key in latent_keys:
                values = [r["latent_metrics"][key] for r in batch_results if key in r["latent_metrics"]]
                if values:
                    aggregated["latent_metrics"][key] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                    }
        
        # Aggregate image metrics
        if batch_results and "image_metrics" in batch_results[0]:
            image_keys = batch_results[0]["image_metrics"].keys()
            for key in image_keys:
                values = [r["image_metrics"][key] for r in batch_results if key in r["image_metrics"]]
                if values:
                    aggregated["image_metrics"][key] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                    }
        
        return aggregated

    def aggregate_rollout_metrics(self, rollout_results):
        """Aggregate rollout metrics"""
        aggregated = {
            "total_rollouts": len(rollout_results),
            "rollout_metrics": {},
        }
        
        if rollout_results and "rollout_metrics" in rollout_results[0]:
            rollout_keys = rollout_results[0]["rollout_metrics"].keys()
            for key in rollout_keys:
                values = [r["rollout_metrics"][key] for r in rollout_results if key in r["rollout_metrics"]]
                if values:
                    aggregated["rollout_metrics"][key] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                    }
        
        return aggregated

    def plot_predictions(self, gt_images, pred_images, reconstructed_images, batch_idx):
        """Generate prediction plots"""
        try:
            from torchvision import utils
            
            num_samples = min(self.cfg.evaluation.num_plot_samples, gt_images.shape[0])
            
            # Sample images
            gt_images, pred_images, reconstructed_images = sample_tensors(
                [gt_images, pred_images, reconstructed_images],
                num_samples,
                indices=list(range(num_samples))
            )
            
            num_frames = gt_images.shape[1]
            
            # Fill in blank images for predictions
            if pred_images is not None:
                pred_images = torch.cat(
                    (
                        torch.full(
                            (num_samples, self.model.num_pred, *pred_images.shape[2:]),
                            -1,
                            device=self.device,
                        ),
                        pred_images,
                    ),
                    dim=1,
                )
            else:
                pred_images = torch.full(gt_images.shape, -1, device=self.device)
            
            # Reshape for plotting
            pred_images = rearrange(pred_images, "b t c h w -> (b t) c h w")
            gt_images = rearrange(gt_images, "b t c h w -> (b t) c h w")
            reconstructed_images = rearrange(reconstructed_images, "b t c h w -> (b t) c h w")
            
            # Concatenate images
            imgs = torch.cat([gt_images, pred_images, reconstructed_images], dim=0)
            
            # Save plot
            plot_path = self.plots_dir / f"predictions_batch_{batch_idx}.png"
            utils.save_image(
                imgs,
                plot_path,
                nrow=num_samples * num_frames,
                normalize=True,
                value_range=(-1, 1),
            )
            
            log.info(f"Saved prediction plot: {plot_path}")
            
        except Exception as e:
            log.warning(f"Failed to generate plots for batch {batch_idx}: {e}")

    def save_results(self, results):
        """Save evaluation results to files"""
        # Save detailed results as JSON
        results_path = self.results_dir / "detailed_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json_results = convert_numpy(results)
        
        with open(results_path, "w") as f:
            json.dump(json_results, f, indent=2)
        
        # Save results as pickle for full fidelity
        pickle_path = self.results_dir / "results.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)
        
        # Create summary report
        self.create_summary_report(results)
        
        log.info(f"Results saved to: {self.results_dir}")

    def create_summary_report(self, results):
        """Create a human-readable summary report"""
        report_path = self.results_dir / "summary_report.txt"
        
        with open(report_path, "w") as f:
            f.write("Model Evaluation Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {self.cfg.model_name}\n")
            f.write(f"Epoch: {self.cfg.model_epoch}\n")
            f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"  Test samples: {results['prediction_results']['total_batches']}\n")
            f.write(f"  Rollout samples: {results['rollout_results']['total_rollouts']}\n\n")
            
            f.write("Overall Loss:\n")
            loss_info = results['prediction_results']['loss']
            f.write(f"  Mean: {loss_info['mean']:.6f}\n")
            f.write(f"  Std:  {loss_info['std']:.6f}\n\n")
            
            if results['prediction_results']['loss_components']:
                f.write("Loss Components:\n")
                for comp, stats in results['prediction_results']['loss_components'].items():
                    f.write(f"  {comp}: {stats['mean']:.6f} ± {stats['std']:.6f}\n")
                f.write("\n")
            
            if results['prediction_results']['latent_metrics']:
                f.write("Latent Space Metrics:\n")
                for metric, stats in results['prediction_results']['latent_metrics'].items():
                    f.write(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}\n")
                f.write("\n")
            
            if results['prediction_results']['image_metrics']:
                f.write("Image Quality Metrics:\n")
                for metric, stats in results['prediction_results']['image_metrics'].items():
                    f.write(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}\n")
                f.write("\n")
            
            if results['rollout_results']['rollout_metrics']:
                f.write("Rollout Metrics:\n")
                for metric, stats in results['rollout_results']['rollout_metrics'].items():
                    f.write(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}\n")
        
        log.info(f"Summary report saved to: {report_path}")

    def run_evaluation(self):
        """Main evaluation pipeline"""
        log.info("Starting model evaluation...")
        
        start_time = time.time()
        
        # Run prediction evaluation
        prediction_results = self.evaluate_predictions()
        
        # Run rollout evaluation
        rollout_results = self.evaluate_rollout()
        
        # Combine results
        final_results = {
            "prediction_results": prediction_results,
            "rollout_results": rollout_results,
            "evaluation_time": time.time() - start_time,
        }
        
        # Save results
        self.save_results(final_results)
        
        # Print summary to console
        self.print_summary(final_results)
        
        log.info("Model evaluation completed successfully")
        return final_results

    def print_summary(self, results):
        """Print evaluation summary to console"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"Model: {self.cfg.model_name}")
        print(f"Epoch: {self.cfg.model_epoch}")
        print(f"Evaluation Time: {results['evaluation_time']:.2f} seconds")
        print()
        
        print("Dataset:")
        print(f"  Test batches: {results['prediction_results']['total_batches']}")
        print(f"  Rollout samples: {results['rollout_results']['total_rollouts']}")
        print()
        
        print("Overall Performance:")
        loss_info = results['prediction_results']['loss']
        print(f"  Loss: {loss_info['mean']:.6f} ± {loss_info['std']:.6f}")
        print()
        
        if results['prediction_results']['latent_metrics']:
            print("Latent Space Metrics:")
            for metric, stats in results['prediction_results']['latent_metrics'].items():
                print(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print()
        
        if results['prediction_results']['image_metrics']:
            print("Image Quality Metrics:")
            for metric, stats in results['prediction_results']['image_metrics'].items():
                print(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print()
        
        print("=" * 60)


@hydra.main(config_path="conf", config_name="eval")
def main(cfg: OmegaConf):
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('eval_results/logs/evaluation.log'),
            logging.StreamHandler()
        ]
    )
    
    log.info("Starting model evaluation...")
    
    try:
        evaluator = ModelEvaluator(cfg)
        results = evaluator.run_evaluation()
        log.info("Evaluation completed successfully")
        
    except Exception as e:
        log.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
