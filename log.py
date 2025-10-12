import os, torch
from torch.utils.tensorboard import SummaryWriter
from config import *
import random, numpy as np

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def save_checkpoint(path, q_net, target, opt, global_frames, last_ckpt_frame, last_sync,
                    train_policy=None, rb=None, extra=None):
    ckpt = {
        "q_net": q_net.state_dict(),
        "target": target.state_dict(),
        "optimizer": opt.state_dict(),
    }
    '''
    ckpt = {
        "q_net": q_net.state_dict(),
        "target": target.state_dict(),
        "optimizer": opt.state_dict(),
        "global_frames": int(global_frames),
        "last_ckpt_frame": int(last_ckpt_frame),
        "last_sync": int(last_sync),
        "rng_cpu": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "config_snapshot": {  
            "SEED": SEED, "GAMMA": GAMMA, "BATCH": BATCH,
            "TRAIN_FREQ": TRAIN_FREQ, "EPS_DECAY": EPS_DECAY, "TARGET_SYNC": TARGET_SYNC,
        },
    }
    if train_policy is not None:
        ckpt["policy"] = train_policy.state_dict()
    if rb is not None:
        if hasattr(rb, "state_dict"):
            ckpt["replay"] = rb.state_dict()

    if extra:
        ckpt["extra"] = extra  # 比如 last_ckpt_frame, last_loss
    '''
    torch.save(ckpt, path)

'''
def load_checkpoint(path, q_net, target, opt, device, train_policy=None, rb=None):
    ckpt = torch.load(path, map_location='cpu')
    q_net.load_state_dict(ckpt["q_net"]); target.load_state_dict(ckpt["target"])
    opt.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng_cpu"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    if "numpy_rng" in ckpt: np.random.set_state(ckpt["numpy_rng"])
    if "python_rng" in ckpt: random.setstate(ckpt["python_rng"])
    if train_policy is not None and "policy" in ckpt:
        train_policy.load_state_dict(ckpt["policy"])
    if rb is not None and "replay" in ckpt and hasattr(rb, "load_state_dict"):
        rb.load_state_dict(ckpt["replay"])
    return ckpt.get("global_frames", 0), ckpt.get("last_ckpt_frame"), ckpt.get("last_sync", 0)
'''


def save_final_weights(path, q_net):
    torch.save(q_net.state_dict(), path)


def make_writer():
    writer = SummaryWriter(LOG_DIR)
    return writer
