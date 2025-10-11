import torch
from config import *
from tqdm.auto import tqdm

def make_opt(q_net):
    opt = torch.optim.RMSprop(
        q_net.parameters(),
        lr=LR, alpha=ALPHA, eps=EPS, momentum=MOMENTUM, centered=False
    )
    return opt

def make_pbar():
    ## 一个帧等于四个原始帧，考虑到结尾，不到4倍
    total_frames = TOTAL_FRAMES*4
    TOTAL = locals().get("TOTAL_FRAMES", total_frames)
    pbar = tqdm(total=TOTAL, desc="Training", unit="frames", dynamic_ncols=True)
    return pbar