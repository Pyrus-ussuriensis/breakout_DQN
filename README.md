# Breakout DQN
## Introduction
在学习完DRL的一些资料后我决定尝试学习写一个网络。开始选择了简单的DQN来解决breakout问题。在阅读了Rainbow等论文和资料后添加上了部分Rainbow的部件。学习项目。
## Network Design
网络设计实现了Rainbow中6个部件中的4个在main中，同时fix-tests中有第5个distributional net的实现，但是由于在实际测试中会增加大量的训练时间，由于训练资源不足没有实际使用。n-steps最终没有添加成功，同时在论文中对于breakout没有益处最终也没有实现。
## Usage
```bash
conda env create -f docs/environment.yml
python -m pip install -r docs/requirements.txt
```

