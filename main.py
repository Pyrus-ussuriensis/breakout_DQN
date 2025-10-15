from config import *
from env import make_env
from policy import make_policy
from collect_data import make_collector
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from log import *
from opt_pbar import *


from torchrl.objectives import DistributionalDQNLoss
from torchrl.modules import DistributionalQValueModule
from tensordict.nn import TensorDictSequential
from torchrl.collectors import SyncDataCollector
from collections import deque
import time, os, math


global_frames = 0 # 记录总的训练的帧数的变量
last_sync = 0 # 记录最后一次同步的帧数
last_ckpt_frame = 0 # 记录最后一次保存权重的帧数   
last_loss_value = 0.0 # 记录最后一次损失，初始值为0.0故开头积累数据时可能一直为0

# 获取其他文件提供的工具
env = make_env(seed=SEED)
q_net, target, train_policy, _, loss_mod = make_policy(env, device)
collector, rb = make_collector(env, train_policy, device)
writer = make_writer()
opt = make_opt(q_net)
pbar = make_pbar()



def train_step(rb, q, target, opt, device, gamma=0.99, batch_size=32):
    q_net.train(); q_net.reset_noise()
    target.eval()
    batch = rb.sample(batch_size).to(device)


    loss_td = loss_mod(batch)      # loss_td["loss"] 形状：[B]（因为 reduction="none"）
    per_sample_grad = loss_td["loss"]

    w = batch.get("_weight", torch.ones_like(per_sample_grad, device=per_sample_grad.device)).to(device) # 如果没有设置_weight则使用后面的默认为1
    #loss = (per_sample_grad * w).mean()
    loss = (per_sample_grad * w).sum() / (w.sum() + 1e-8)


    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
    opt.step()

    per_sample = loss_td["loss"].detach().abs()      # 形状: [B]

    prio = per_sample.to("cpu").view(-1)             # 要一维 [B]
    info = TensorDict(
        {"index": batch["index"].detach().cpu(),
        "td_error": prio},
        batch_size=[prio.shape[0]],
    )
    rb.update_tensordict_priority(info)


    return float(loss.detach().cpu().item())



for batch in collector:
    T, P  = batch.batch_size
    T_eff = batch["next", "reward"].shape[0]
    if T_eff == 0:
        continue
    tp = lambda x: x[:T_eff].flatten(0, 1)

    obs      = tp(batch["pixels"]).to(torch.uint8)
    next_obs = tp(batch["next","pixels"]).to(torch.uint8)
    action = tp(batch["action"])
    reward = tp(batch["next","reward"]); 
    done   = tp(batch["next","done"])
    action = action.squeeze(-1).long()
    reward = reward.view(-1,1).float()
    done = done.view(-1,1).bool()

    N = obs.shape[0]
    td_tr = TensorDict({
        "pixels": obs,
        "action": action,
        ("next", "pixels"): next_obs,
        ("next", "reward"): reward,
        ("next", "done"): done,
    }, batch_size=[N])
    rb.extend(td_tr)

    global_frames += N
    pbar.update(N) 

    if global_frames > LEARN_STARTS:
        updates = max(1, N // TRAIN_FREQ)
        for _ in range(updates):
            last_loss_value = train_step(rb, q_net, target, opt, device, gamma=GAMMA, batch_size=BATCH)

    if global_frames - last_sync >= TARGET_SYNC:
        k = (global_frames - last_sync) // TARGET_SYNC
        target.load_state_dict(q_net.state_dict())
        last_sync += TARGET_SYNC*k

    if global_frames - last_ckpt_frame >= CKPT_EVERY_FRAMES:
        save_checkpoint(os.path.join(CKPT_DIR, f"ckpt_{global_frames}.pt"),
                        q_net, target, opt, global_frames, last_ckpt_frame, last_sync, train_policy=train_policy, rb=rb)
        last_ckpt_frame = global_frames

    try:
        replay_size = len(rb)
    except TypeError:
        replay_size = getattr(rb._storage, "_cursor", 0)

    writer.add_scalar("train/loss", last_loss_value, global_frames)
    writer.add_scalar("train/replay_size", replay_size, global_frames)
    writer.add_scalar("train/reward_mean", reward.float().mean().item(), global_frames)
    writer.add_scalar("train/reward_nonzero", (reward!=0).float().mean().item(), global_frames)


pbar.close()
writer.flush(); writer.close()

save_final_weights(os.path.join(CKPT_DIR, "q_net_final.pt"), q_net)
torch.save(target.state_dict(), os.path.join(CKPT_DIR, "target_final.pt"))

