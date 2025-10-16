from config import *
from env import make_env
from policy import make_policy
from collect_data import make_collector
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from log import *
from opt_pbar import *
from torchrl.collectors import SyncDataCollector
from collections import deque
import time, os, math


global_frames = 0 # 记录总的训练的帧数的变量
last_sync = 0 # 记录最后一次同步的帧数
last_ckpt_frame = 0 # 记录最后一次保存权重的帧数   
last_loss_value = 0.0 # 记录最后一次损失，初始值为0.0故开头积累数据时可能一直为0

# 获取其他文件提供的工具
env = make_env(seed=SEED)
q_net, target, train_policy, _ = make_policy(env, device)
collector, rb = make_collector(env, train_policy, device)
writer = make_writer()
opt = make_opt(q_net)
pbar = make_pbar()



# 训练主流程
def train_step(rb, q, target, opt, device, gamma=0.99, batch_size=32):
    batch = rb.sample(batch_size)

    s  = batch["obs"].to(device)
    a  = batch["action"].to(device)
    r  = batch["reward"].to(device)
    s2 = batch["next_obs"].to(device)
    d  = batch["done"].to(device)

    a   = a.long().view(-1, 1) 
    qsa = q(s).gather(1,a).squeeze(1)
    with torch.no_grad():
        a2 = q(s2).argmax(1, keepdim=True)
        q2 = target(s2).gather(1,a2).squeeze(-1)
        y  = r + (1.0 - d.float()) * gamma * q2

    #loss = F.smooth_l1_loss(qsa, y)
    per_sample = torch.nn.functional.smooth_l1_loss(qsa, y, reduction="none")

    w = batch.get("_weight", torch.ones_like(per_sample, device=per_sample.device)).to(device) # 如果没有设置_weight则使用后面的默认为1
    loss = (per_sample * w).mean()



    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
    opt.step()

    with torch.no_grad():
        td_err = (qsa - y).abs().detach().cpu()
    batch.set("td_error", td_err)  # 给tensordict设置一个参数td_error和action等是并列的。priority_key='td_error'
    rb.update_tensordict_priority(batch)

    return float(loss.item())


# 收集数据流程
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
    reward = reward.squeeze(-1).float()
    done = done.squeeze(-1).bool()

    N = obs.shape[0]
    td_tr = TensorDict({
        "obs":      obs,
        "action":   action,
        "reward":   reward,
        "next_obs": next_obs,
        "done":     done,
    }, batch_size=[N])
    rb.extend(td_tr)

    global_frames += N
    pbar.update(N) 

    # 开始收集数据不优化网络，到达一定数量后按照每次收集的数据的量更新网络
    if global_frames > LEARN_STARTS:
        updates = max(1, N // TRAIN_FREQ)
        for _ in range(updates):
            last_loss_value = train_step(rb, q_net, target, opt, device, gamma=GAMMA, batch_size=BATCH)

    # 多少帧间隔后同步网络权重
    if global_frames - last_sync >= TARGET_SYNC:
        k = (global_frames - last_sync) // TARGET_SYNC
        target.load_state_dict(q_net.state_dict())
        last_sync += TARGET_SYNC*k

    # 间隔多少帧后保存权重
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

