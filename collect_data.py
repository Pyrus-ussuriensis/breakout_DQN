from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import LazyMemmapStorage, ReplayBuffer
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

    rb = ReplayBuffer(storage=LazyMemmapStorage(max_size=REPLAY_CAP, scratch_dir=BUFFER_DIR))
    try:
        rb = ReplayBuffer(storage=LazyMemmapStorage(max_size=REPLAY_CAP, scratch_dir=BUFFER_DIR))
    except FileExistsError:
        shutil.rmtree(BUFFER_DIR, ignore_errors=True)
        os.makedirs(BUFFER_DIR, exist_ok=True)
        rb = ReplayBuffer(storage=LazyMemmapStorage(max_size=REPLAY_CAP, scratch_dir=BUFFER_DIR))

    return collector, rb