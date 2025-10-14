from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.envs.transforms import MultiStepTransform
import os, shutil
from config import *

# 产生训练数据，自动调用策略在环境中训练，自动进行格式转化，将动作转化为Numpy送入Env
def make_collector(env, train_policy, device):
    collector = SyncDataCollector(
        env, policy=train_policy,
        policy_device=device,
        env_device='cpu',
        storing_device='cpu',
        frames_per_batch=1024,      # 每批收多少帧
        total_frames=TOTAL_FRAMES,     # 总帧数上限
        split_trajs=True            # 让终止切开
    )

    sampler = PrioritizedSampler(
        max_capacity=REPLAY_CAP, # 采样器最大容量
        alpha=0.6,      # 优先级强度，越大越采样损失大的，易过拟合
        beta=0.4,       # 重要性采样起始，校正
        eps=1e-6, # offset
        max_priority_within_buffer=True # 新样本最大优先级
    )

    shutil.rmtree(BUFFER_DIR, ignore_errors=True)
    os.makedirs(BUFFER_DIR, exist_ok=True)
    rb = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(max_size=REPLAY_CAP, scratch_dir=BUFFER_DIR),
        sampler=sampler,
        batch_size=BATCH,   
        # pin_memory=False, prefetch=None  
        priority_key="td_error"  # 缺省是 'td_error'
        #transform=MultiStepTransform(n_steps=N_STEP, gamma=GAMMA)
    )
    #rb = ReplayBuffer(storage=LazyMemmapStorage(max_size=REPLAY_CAP, scratch_dir=BUFFER_DIR))#, pin_memory=False, prefetch=2, batch_size=BATCH)

    return collector, rb

