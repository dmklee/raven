from typing import Optional, Tuple, List, Dict
from pathlib import Path

import numpy as np
from tqdm import tqdm
import os
from filelock import FileLock
from threadpoolctl import threadpool_limits
import shutil
import zarr
import h5py
import concurrent.futures
import multiprocessing

import torch
from torch import Tensor
from torch.utils.data import Dataset

from uferl.data.transform import JointTransform
from uferl.data.types import Obs, Action
from uferl.common.pytorch_util import dict_apply
from uferl.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from uferl.model.common.rotation_transformer import RotationTransformer
from uferl.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from uferl.common.replay_buffer import ReplayBuffer
from uferl.common.sampler import SequenceSampler, get_val_mask
from uferl.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat2,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


class RobomimicReplayImageDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        cache_path: str,
        shape_meta,
        horizon,
        n_obs_steps,
        pad_before,
        pad_after,
        num_demos: Optional[int]=None,
        # transform: Optional[JointTransform]=None,
        abs_action: bool=True,
        rotation_rep='rotation_6d', # ignored when abs_action=False
        use_cache: bool = True,
        val_ratio: float=0.0,
        use_legacy_normalizer: bool=False,
        seed: int=42,
    ):
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None
        if use_cache:
            cache_zarr_path = cache_path + f'.{num_demos}.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_path=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                            n_demo=num_demos,
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
                n_demo=num_demos
            )

        rgb_keys = list()
        depth_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            if type == 'depth':
                depth_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + depth_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.depth_keys = depth_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

        # self.transform = transform

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys + self.depth_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            # if 'extrinsic' in key:
                # v =  np.max(data[key][T_slice]) <= 1 and np.min(data[key][T_slice]) >= -1
                # if not v:
                    # print(data[key][T_slice])
                    # print(np.max(data[key][T_slice]), np.min(data[key][T_slice]))
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat2(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # relative action 
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            quat2mtx = RotationTransformer(from_rep='quaternion', to_rep='matrix')

            n_arms = 2 if stat['mean'].shape[-1] > 10 else 1
            action_dim = stat['mean'].shape[-1] // n_arms

            min_pos = np.zeros((n_arms, 3))
            max_pos = np.zeros((n_arms, 3))
            for i in range(self.replay_buffer.meta.episode_ends.shape[0]):
                if i == 0:
                    start = 0
                else:
                    start = self.replay_buffer.meta.episode_ends[i-1]
                end = self.replay_buffer.meta.episode_ends[i]
                for horizon in range(self.horizon):
                    abs_action = self.replay_buffer['action'][start + horizon:end]
                    abs_action = abs_action.reshape(abs_action.shape[0], n_arms, -1)
                    abs_pos_homog = np.concatenate([
                        abs_action[..., :3],
                        np.ones((*abs_action.shape[:-1], 1)),
                    ], axis=-1)

                    eef_pose = np.tile(np.eye(4), (abs_action.shape[0], n_arms, 1, 1))
                    for arm_id in range(n_arms):
                        eef_pose[:, arm_id, :3, 3] = self.replay_buffer[f'robot{arm_id}_eef_pos'][start:end-horizon]
                        eef_pose[:, arm_id, :3, :3] = quat2mtx.forward(
                            self.replay_buffer[f'robot{arm_id}_eef_quat'][start:end-horizon][:, [3, 0, 1, 2]]
                        )
                    tfm = np.linalg.inv(eef_pose)
                    rel_pos = np.einsum('naij, naj->nai', tfm[..., :3, :], abs_pos_homog)

                    min_pos = np.minimum(min_pos, rel_pos.min(0))
                    max_pos = np.maximum(max_pos, rel_pos.max(0))

            for a in range(n_arms):
                stat['min'][action_dim*a:action_dim*a+3] = min_pos[a]
                stat['max'][action_dim*a:action_dim*a+3] = max_pos[a]
                stat['mean'][action_dim*a:action_dim*a+3] = 0.

            if n_arms == 1:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat2(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['rel_action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('world2pix') or key.endswith('extrinsic') or key.endswith('intrinsic'):
                # dont want to normalize
                this_normalizer = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] > 10:
            # dual arm
            raw_actions = raw_actions.reshape(raw_actions.shape[0], 2, -1)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(raw_actions.shape[0], -1)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer=None,
    n_workers=None,
    max_inflight_tasks=None,
    n_demo=None,
):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    n_workers = 3

    # # parse shape_meta
    rgb_keys = list()
    depth_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        _type = attr.get('type', 'low_dim')
        if _type == 'rgb':
            rgb_keys.append(key)
        elif _type == 'depth':
            depth_keys.append(key)
        elif _type == 'low_dim':
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        demos = file['data']
        if n_demo is None:
            n_demo = len(demos)
        # count total steps
        episode_ends = list()
        prev_end = 0
        max_len = 0
        for i in range(n_demo):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            max_len = max(episode_length, max_len)
            prev_end = episode_end
            episode_ends.append(episode_end)
        # print(max_len)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends,
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(n_demo):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape']), 'action shape mismatch {}'.format(this_data.shape[1:])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape']), 'obs[{}] shape mismatch: {}'.format(key, this_data.shape[1:])

            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )

        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                raise e
                return False

        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in depth_keys + rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(n_demo):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        if key.endswith('depth'):
                            float_arr = (hdf5_arr[...] * 255).astype(np.uint8)
                            hdf5_arr = float_arr
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy,
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


if __name__ == "__main__":
    shape_meta = {
        'obs' : {
            'agentview_image' : dict(shape=[3, 84, 84], type='rgb'),
            'robot0_eye_in_hand_image' : dict(shape=[3, 84, 84], type='rgb'),
            'robot0_eef_pos' : dict(shape=[3]),
            'robot0_eef_quat' : dict(shape=[4]),
            'robot0_gripper_qpos' : dict(shape=[2]),
        },
        'action' : {
            'shape': [10]
        }
    }
    d = RobomimicReplayImageDataset(
        dataset_path='data/robomimic/stack_d1_obs_abs.hdf5',
        cache_path='tmp',
        shape_meta=shape_meta,
        horizon=16,
        n_obs_steps=1,
        pad_before=0,
        pad_after=7,
        num_demos=10,
        use_cache=False,
    )
    d.get_normalizer()
    exit()
    for k, v in a['obs'].items():
        print(k, v.shape)
    print('action', a['action'].shape)
