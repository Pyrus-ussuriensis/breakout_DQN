import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import QValueModule, EGreedyModule, NoisyLinear
from config import *

# 根据观察给出动作价值的网络
class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )

        self.val = nn.Sequential(
            NoisyLinear(64*7*7, 512), nn.ReLU(inplace=True),
            NoisyLinear(512, 1),      # V(s)
        )
        self.adv = nn.Sequential(
            NoisyLinear(64*7*7, 512), nn.ReLU(inplace=True),
            NoisyLinear(512, n_actions),  # A(s,a)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.float() / 255.0
        h = self.net(x)                  # (B, 64*7*7)
        V = self.val(h)                   # (B, 1)
        A = self.adv(h)                   # (B, nA)
        A_centered = A - A.mean(dim=1, keepdim=True)   # 约束：∑A=0
        Q = V + A_centered                # (B, nA)
        return Q
    
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

class NoisyWrapper:
    def __init__(self, greedy_policy, q_net, k=1):
        self.greedy = greedy_policy        # TensorDictSequential(actor, qvalue)
        self.q = q_net
        self.k = int(k)
        self.t = 0

    def __call__(self, td):
        if self.t % self.k == 0:
            self.q.train()
            self.q.reset_noise()     
        self.t += 1
        return self.greedy(td)

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

    qvalue = QValueModule(
        spec=env.action_spec,
        action_space="categorical",
        action_value_key="action_value",
        out_keys=["action", "action_value"],
        safe=True,
    )

    greedy = TensorDictSequential(actor, qvalue).to(device)
    train_policy = NoisyWrapper(greedy, q_net, k=4) # 包装后每隔几步会重新调整噪声，代替了epsilon
    def eval_policy(td):
        q_net.eval()
        return greedy(td)

    

    return q_net, target, train_policy, eval_policy
