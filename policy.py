import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import QValueModule, EGreedyModule, NoisyLinear
from torchrl.modules import DistributionalQValueModule
from torchrl.objectives import DistributionalDQNLoss
from torchrl.modules.tensordict_module import DistributionalQValueActor
from config import *

# 根据观察给出动作的网络
class DQN(nn.Module):
    def __init__(self, n_actions, n_atoms=51, vmin=-10.0, vmax=10.0):
        super().__init__()
        self.nA, self.N = n_actions, n_atoms
        self.register_buffer("support", torch.linspace(vmin, vmax, n_atoms))
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )

        self.val = nn.Sequential(
            NoisyLinear(64*7*7, 512), nn.ReLU(inplace=True),
            NoisyLinear(512, self.N), # V(s)
        )
        self.adv = nn.Sequential(
            NoisyLinear(64*7*7, 512), nn.ReLU(inplace=True),
            NoisyLinear(512, n_actions*self.N), # A(s,a)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.float() / 255.0
        h = self.net(x) # (B, 64*7*7)
        V = self.val(h).unsqueeze(-1) # (B, N, 1)
        A = self.adv(h).view(-1, self.N, self.nA) # (B, N, nA)
        A = A - A.mean(dim=-1, keepdim=True) 
        Q = V + A # (B, N, nA)
        return Q
    
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

class NoisyWrapper:
    def __init__(self, greedy_policy, q_net, k=1):
        self.greedy = greedy_policy # TensorDictSequential(actor, qvalue)
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

    class DistExpect(torch.nn.Module):
        def __init__(self, support): super().__init__(); self.register_buffer("support", support.view(1,-1,1).float())
        def forward(self, param):    # param: [B, N, nA] (logits)
            probs = param.softmax(1)
            return (probs * self.support).sum(dim=1)  # [B, nA]

    expect_head = TensorDictModule(DistExpect(q_net.support), in_keys=["action_value"], out_keys=["action_value"])

    qvalue = QValueModule(
        spec=env.action_spec,
        action_space="categorical",
        action_value_key="action_value",
        out_keys=["action", "action_value"],
        safe=True,
    )

    # 贪心头
    greedy = TensorDictSequential(actor, expect_head, qvalue).to(device)
    train_policy = NoisyWrapper(greedy, q_net, k=1)
    def eval_policy(td):
        q_net.eval()
        return greedy(td)


    actor_loss = TensorDictModule(
        q_net,
        in_keys=["pixels"],
        out_keys=["distr_logits"],
    )
    value_net = DistributionalQValueActor(
        module=actor_loss,
        in_keys="pixels",
        support=q_net.support,
        action_space="categorical",
        action_value_key="distr_logits",
        make_log_softmax=True
    )
    # 损失头
    loss_mod = DistributionalDQNLoss(
        value_network=value_net,  # ← 不要传 q_net；传到 action_value 这一层即可
        gamma=0.99, delay_value=True, reduction="none",
    )
    loss_mod.set_keys(action_value="distr_logits")

    return q_net, target, train_policy, eval_policy, loss_mod
