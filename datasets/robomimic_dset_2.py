import torch
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from einops import rearrange
from .traj_dset import TrajDataset, TrajSlicerDataset, get_train_val_sliced


class RoboMimicDataset(TrajDataset):
    def __init__(
        self,
        config,
        split="train",
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        normalize_state: bool = True,
        n_rollout: Optional[int] = None,
    ):
        self.config = config
        self.split = split
        self.transform = transform
        self.normalize_action = normalize_action
        self.normalize_state = normalize_state
        self.n_rollout = n_rollout
        
        # Load data using existing RoboMimic utilities
        self._load_data()
        
        # Compute dimensions
        self._compute_dimensions()
        
        # Setup normalization
        self._setup_normalization()
        
        # Apply normalization
        self._apply_normalization()
        
        print(f"Loaded {len(self.episodes)} {split} episodes")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}, Proprio dim: {self.proprio_dim}")

    def _load_data(self):
        """Load RoboMimic data using existing utilities."""
        from environments.robomimic.utils import get_train_val_datasets
        
        # Load train and val episodes
        train_eps, val_eps, norm_dict, state_dim, action_dim = get_train_val_datasets(self.config)
        
        # Select appropriate split
        if self.split == "train":
            self.episodes = train_eps
        elif self.split == "val":
            self.episodes = val_eps
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val'")
        
        # Store dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Limit episodes if requested
        if self.n_rollout is not None:
            episode_keys = list(self.episodes.keys())[:self.n_rollout]
            self.episodes = {k: self.episodes[k] for k in episode_keys}

    def _compute_dimensions(self):
        """Compute dimensions from loaded data."""
        # Get first episode to determine dimensions
        first_ep_key = list(self.episodes.keys())[0]
        first_ep = self.episodes[first_ep_key]
        
        # Compute sequence lengths
        self.seq_lengths = []
        for ep_key in self.episodes.keys():
            ep_length = len(self.episodes[ep_key]["state"])
            self.seq_lengths.append(ep_length)
        
        # Determine proprioceptive dimension from state
        # For RoboMimic, proprioceptive data is typically joint positions/velocities
        # We'll use a subset of the state as proprioceptive
        if hasattr(self.config, 'proprio_dim') and self.config.proprio_dim is not None:
            self.proprio_dim = self.config.proprio_dim
        else:
            # Default: use first 7 dimensions as proprioceptive (common for robot arms)
            self.proprio_dim = min(7, self.state_dim)

    def _setup_normalization(self):
        """Setup normalization parameters."""
        if self.normalize_action or self.normalize_state:
            # Collect all data for computing statistics
            all_actions = []
            all_states = []
            all_proprios = []
            
            for ep_key in self.episodes.keys():
                ep = self.episodes[ep_key]
                all_actions.append(ep["action"])
                all_states.append(ep["state"])
                # Extract proprioceptive data from state
                proprio_data = ep["state"][:, :self.proprio_dim]
                all_proprios.append(proprio_data)
            
            # Concatenate all data
            all_actions = np.concatenate(all_actions, axis=0)
            all_states = np.concatenate(all_states, axis=0)
            all_proprios = np.concatenate(all_proprios, axis=0)
            
            # Compute statistics
            if self.normalize_action:
                self.action_mean = torch.tensor(np.mean(all_actions, axis=0), dtype=torch.float32)
                self.action_std = torch.tensor(np.std(all_actions, axis=0) + 1e-6, dtype=torch.float32)
            else:
                self.action_mean = torch.zeros(self.action_dim)
                self.action_std = torch.ones(self.action_dim)
            
            if self.normalize_state:
                self.state_mean = torch.tensor(np.mean(all_states, axis=0), dtype=torch.float32)
                self.state_std = torch.tensor(np.std(all_states, axis=0) + 1e-6, dtype=torch.float32)
            else:
                self.state_mean = torch.zeros(self.state_dim)
                self.state_std = torch.ones(self.state_dim)
            
            # Proprioceptive normalization
            self.proprio_mean = torch.tensor(np.mean(all_proprios, axis=0), dtype=torch.float32)
            self.proprio_std = torch.tensor(np.std(all_proprios, axis=0) + 1e-6, dtype=torch.float32)
        else:
            # No normalization
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

    def _apply_normalization(self):
        """Apply normalization to loaded data."""
        # Convert episodes to normalized format
        normalized_episodes = {}
        
        for ep_key in self.episodes.keys():
            ep = self.episodes[ep_key]
            
            # Normalize actions and states
            normalized_actions = (ep["action"] - self.action_mean.numpy()) / self.action_std.numpy()
            normalized_states = (ep["state"] - self.state_mean.numpy()) / self.state_std.numpy()
            
            # Create normalized episode
            normalized_ep = {
                "action": torch.tensor(normalized_actions, dtype=torch.float32),
                "state": torch.tensor(normalized_states, dtype=torch.float32),
                "reward": torch.tensor(ep["reward"], dtype=torch.float32),
                "success": torch.tensor(ep["success"], dtype=torch.bool),
                "is_first": torch.tensor(ep["is_first"], dtype=torch.bool),
                "is_last": torch.tensor(ep["is_last"], dtype=torch.bool),
                "is_terminal": torch.tensor(ep["is_terminal"], dtype=torch.bool),
            }
            
            # Add image observations if present
            for key in ep.keys():
                if "image" in key.lower():
                    # Convert images to tensors and normalize
                    images = ep[key]
                    if isinstance(images, np.ndarray):
                        images = torch.tensor(images, dtype=torch.float32)
                    # Normalize to [0, 1] if not already
                    if images.max() > 1.0:
                        images = images / 255.0
                    normalized_ep[key] = images
            
            normalized_episodes[ep_key] = normalized_ep
        
        self.episodes = normalized_episodes

    def get_seq_length(self, idx: int) -> int:
        """Return the length of the idx-th trajectory."""
        return self.seq_lengths[idx]

    def get_all_actions(self):
        """Get all actions from all episodes."""
        result = []
        for ep_key in self.episodes.keys():
            ep = self.episodes[ep_key]
            result.append(ep["action"])
        return torch.cat(result, dim=0)

    def get_frames(self, idx: int, frames):
        """Get specific frames from an episode."""
        ep_key = list(self.episodes.keys())[idx]
        ep = self.episodes[ep_key]
        
        # Extract frames
        actions = ep["action"][frames]
        states = ep["state"][frames]
        rewards = ep["reward"][frames]
        successes = ep["success"][frames]
        
        # Extract proprioceptive data
        proprio = states[:, :self.proprio_dim]
        
        # Prepare observations dictionary
        obs = {"proprio": proprio}
        
        # Add image observations if present
        for key in ep.keys():
            if "image" in key.lower():
                images = ep[key][frames]
                if self.transform:
                    images = self.transform(images)
                obs[key] = images
        
        # Prepare environment info
        env_info = {
            "success": successes,
            "reward": rewards,
        }
        
        return obs, actions, states, env_info

    def __getitem__(self, idx: int):
        """Get full episode data."""
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self) -> int:
        """Return number of episodes."""
        return len(self.episodes)

    def preprocess_imgs(self, imgs):
        """Preprocess images for training."""
        if isinstance(imgs, np.ndarray):
            imgs = torch.tensor(imgs, dtype=torch.float32)
        
        if isinstance(imgs, torch.Tensor):
            # Ensure proper format: (T, C, H, W)
            if imgs.dim() == 4 and imgs.shape[-1] == 3:  # (T, H, W, C)
                imgs = rearrange(imgs, "T H W C -> T C H W")
            # Normalize to [0, 1] if needed
            if imgs.max() > 1.0:
                imgs = imgs / 255.0
        
        return imgs


def load_robomimic_slice_train_val(
    config,
    transform=None,
    n_rollout_train=None,
    n_rollout_val=None,
    normalize_action=True,
    normalize_state=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    # Load train dataset
    train_dset = RoboMimicDataset(
        config=config,
        split="train",
        transform=transform,
        normalize_action=normalize_action,
        normalize_state=normalize_state,
        n_rollout=n_rollout_train,
    )
    
    # Load val dataset
    val_dset = RoboMimicDataset(
        config=config,
        split="val",
        transform=transform,
        normalize_action=normalize_action,
        normalize_state=normalize_state,
        n_rollout=n_rollout_val,
    )
    
    # Create sliced datasets
    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)
    
    # Return in expected format
    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset
    
    return datasets, traj_dset



def test_datset(config):
    pass



if __name__ == "__main__":
    main()