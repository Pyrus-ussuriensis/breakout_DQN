import os, time, torch
import numpy as np
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import QValueModule

from env import make_env       
from policy import DQN          

@torch.no_grad()
def rollout_and_record(weights_path,
                       episodes=3,
                       seed=0,
                       device=None,
                       video_dir="./videos"):
    os.makedirs(video_dir, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(seed=seed, record_video=True, video_dir=video_dir) 
    env.to(device)

    nA = env.action_spec.space.n
    q_net = DQN(nA).to(device).eval()
    sd = torch.load(weights_path, map_location=device)
    state_dict = sd if isinstance(sd, dict) and "q_net" not in sd else sd["q_net"]
    q_net.load_state_dict(state_dict)

    actor  = TensorDictModule(q_net, in_keys=["pixels"], out_keys=["action_value"]).to(device)
    greedy = TensorDictSequential(
        actor,
        QValueModule(spec=env.action_spec,
                     action_value_key="action_value",
                     out_keys=["action","action_value"],
                     safe=True)
    ).to(device)

    returns = []
    for ep in range(episodes):
        td = env.reset()
        ep_ret = 0.0
        while True:
            td = greedy(td)        

            td = env.step(td)     
            ep_ret += float(td["next", "reward"].item())

            done = bool(td["next","done"].item())
            td = td["next"].exclude("reward","done","terminated","truncated")
            if done:
                returns.append(ep_ret)
                print(f"[Eval] episode {ep+1}/{episodes} return={ep_ret:.1f}")
                break

    env.close()
    print("returns:", returns)
    print(f"videos are saved at: {os.path.abspath(video_dir)}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rollout_and_record("./checkpoints/ckpt_802143.pt",
                       episodes=10, seed=0, device=device, video_dir="./videos")
