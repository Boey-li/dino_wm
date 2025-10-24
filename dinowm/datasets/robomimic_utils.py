import numpy as np
from pathlib import Path
import robomimic.utils.file_utils as FileUtils


STATE_SHAPE_META = {
    # "robot0_joint_pos_cos": {
    #     "shape": [7],
    #     "type": "low_dim",
    # },
    # "robot0_joint_pos_sin": {
    #     "shape": [7],
    #     "type": "low_dim",
    # },
    "robot0_eef_pos": {
        "shape": [3],
        "type": "low_dim",
    },
    "robot0_eef_quat": {
        "shape": [4],
        "type": "low_dim",
    },
    "robot0_gripper_qpos": {
        "shape": [2],
        "type": "low_dim",
    },
}

def create_shape_meta(img_size, include_state):
    shape_meta = {
        "obs": {
            "agentview_image": {
                # gym expects (H, W, C)
                "shape": [img_size, img_size, 3],
                "type": "rgb",
            },
            "robot0_eye_in_hand_image": {
                # gym expects (H, W, C)
                "shape": [img_size, img_size, 3],
                "type": "rgb",
            },
        },
        "action": {"shape": [7]},
    }
    if include_state:
        shape_meta["obs"].update(STATE_SHAPE_META)
    return shape_meta

def get_robomimic_dataset_path_and_env_meta(
    env_id,
    collection_type="ph",
    obs_type="image",
    shaped=False,
    image_size=128,
    done_mode=0,
    datadir="/home/dreamerv3/robomimic_datasets",  # Default for docker/singularity
):
    """
    Returns the path to the Robomimic dataset and environment metadata.

    Args:
        env_id (str): The ID of the environment.
        collection_type (str, optional): The type of data collection. Defaults to "ph".
        obs_type (str, optional): The type of observations. Defaults to "image".
        shaped (bool, optional): Whether the dataset is shaped or not. Defaults to False.
        image_size (int, optional): The size of the images in the dataset. Defaults to 128.

    Returns:
        tuple: A tuple containing the dataset path and environment metadata.
    """
    assert int(done_mode) in [0, 1, 2]

    dataset_name = obs_type
    if image_size != 0:
        dataset_name += f"_{image_size}"
    if shaped:
        dataset_name += "_shaped"
    dataset_name += f"_done{done_mode}"
    dataset_path = f"{env_id.lower()}/{collection_type}/{dataset_name}_v141.hdf5"

    root_dir = datadir
    dataset_path = Path(root_dir, dataset_path)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    return dataset_path, env_meta

def get_dataset_path_and_meta_info(
    env_id,
    collection_type="ph",
    obs_type="image",
    shaped=False,
    image_size=128,
    done_mode=0,
    datadir="/home/dreamerv3/robomimic_datasets",
):
    """
    Returns the path to the Robomimic dataset and environment metadata.

    Args:
        env_id (str): The ID of the environment.
        collection_type (str, optional): The type of data collection. Defaults to "ph".
        obs_type (str, optional): The type of observations. Defaults to "image".
        shaped (bool, optional): Whether the dataset is shaped or not. Defaults to False.
        image_size (int, optional): The size of the images in the dataset. Defaults to 128.

    Returns:
        tuple: A tuple containing the dataset path and environment metadata.
    """
    dataset_path, env_meta = get_robomimic_dataset_path_and_env_meta(
        env_id,
        collection_type=collection_type,
        obs_type=obs_type,
        shaped=shaped,
        image_size=image_size,
        done_mode=done_mode,
        datadir=datadir,
    )
    shape_meta = create_shape_meta(image_size, include_state=True)
    return dataset_path, env_meta, shape_meta


def add_traj_to_cache(demo, f, pixel_keys, state_keys, norm_dict=None):
    traj = f["data"][demo]
    
    # Create "state" key
    concat_state = []
    for t in range(len(traj["obs"][pixel_keys[0]])):
        if 'state' in traj['obs'].keys():
            curr_obs_state_vec = traj["obs"]['state'][t]
        else:
            curr_obs_state_vec = [traj["obs"][obs_key][t] for obs_key in state_keys]
            curr_obs_state_vec = np.concatenate(curr_obs_state_vec, dtype=np.float32)
        concat_state.append(curr_obs_state_vec)

        # Update norm_dict for the states
        if norm_dict is not None:
            norm_dict["ob_max"] = np.maximum(norm_dict["ob_max"], curr_obs_state_vec)
            norm_dict["ob_min"] = np.minimum(norm_dict["ob_min"], curr_obs_state_vec)
    states = np.stack(concat_state, axis=0)  # (T, state_dim)
    seq_length = states.shape[0]
    
    # Create image obs keys
    obs_imgs = {}
    for key in pixel_keys:
        obs_imgs[key] = np.array(traj["obs"][key][:])  # (T, H, W, C)
        assert obs_imgs[key].shape[0] == seq_length, f"Image length {obs_imgs[key].shape[0]} does not match state length {seq_length}"
    
    # Create action keys
    actions = np.array(traj["actions"][:])  # (T, action_dim)
    assert actions.shape[0] == seq_length, f"Action length {actions.shape[0]} does not match state length {seq_length}"
    
    # update norm_dict for actions
    if norm_dict is not None:
        norm_dict["ac_max"] = np.maximum(norm_dict["ac_max"], np.max(actions, axis=0))
        norm_dict["ac_min"] = np.minimum(norm_dict["ac_min"], np.min(actions, axis=0))
    
    return states, actions, obs_imgs, seq_length, norm_dict
    
    
    