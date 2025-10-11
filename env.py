from gymnasium.wrappers import RecordVideo
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import ToTensorImage, GrayScale, Resize, CatFrames, Compose
import os
import gymnasium as gym
import torch

from torchrl.envs.libs.gym import GymWrapper
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# 创建环境
def make_env(seed=0, record_video=False, video_dir=None, video_trigger=None, eval_mode=False):
    kwargs = {}
    # 若开始预测，需要录制视频
    if record_video:
        kwargs["render_mode"] = "rgb_array"
    if eval_mode:
        kwargs["repeat_action_probability"] = 0.0
    # 这里需要设置frameskip=1，不然和下面的堆叠重复会影响训练效果
    env = gym.make("ALE/Breakout-v5", frameskip=1, **kwargs)
    
    # 按照cleanrl的包装简单修改
    # 需要放到最后，才能录制多个视频，因为后面的只要reset就会影响前面的
    # 如果放在前面EpisodicLifeEnv能够吞掉一般的reset，能够录制一个长视频，但是真的结束时仍然会结束
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(env, video_folder=video_dir,
                          episode_trigger=(video_trigger or (lambda ep: True)),
                          #step_trigger=lambda ep: ep == 0,  # 一开始就录
                          #video_length=10_000_000,              # 足够长
                          name_prefix="eval_all")
     
    
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
    #env = gym.wrappers.FrameStack(env, 4)

                         
    env = GymWrapper(env, categorical_action_encoding=True)
    env = TransformedEnv(env, Compose(
        ToTensorImage(in_keys=["observation"], out_keys=["pixels"], from_int=False, dtype=torch.float32),
        CatFrames(N=4, dim=-3, in_keys=["pixels"], out_keys=["pixels"])
    ))
    env.reset(seed=seed); env.action_space.seed(seed)
    return env
