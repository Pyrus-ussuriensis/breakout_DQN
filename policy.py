import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import QValueModule, EGreedyModule
from config import *

# 根据观察给出动作的网络
class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)                  # (1,4,84,84)
        x = x.float()/255
        x = self.net(x)                         # (B,64,7,7)
        x = x.flatten(start_dim=1)
        return self.head(x)                     # (B, |A|)

# 给出策略
def make_policy(env, device):
    n_actions = env.action_spec.space.n

    # 训练网络和目标网络
    q_net  = DQN(n_actions).to(device)
    target = DQN(n_actions).to(device)
    target.load_state_dict(q_net.state_dict())

    actor = TensorDictModule(
        q_net,
        in_keys=["pixels"],
        out_keys=["action_value"],
    )

    greedy = TensorDictSequential(
        actor,
        QValueModule(
            spec=env.action_spec,
            action_space="categorical",
            action_value_key="action_value",
            out_keys=["action", "action_value"],  
            safe=True,
        ),
    ).to(device)

    train_policy = TensorDictSequential(
        greedy,
        EGreedyModule(
            spec=env.action_spec,
            action_key="action",
            eps_init=EPS_INIT, eps_end=EPS_END,
            annealing_num_steps=EPS_DECAY,
        ),
    ).to(device)

    return q_net, target, train_policy
