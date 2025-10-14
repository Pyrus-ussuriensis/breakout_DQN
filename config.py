import torch
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data 
# 总的训练的帧数限制，通过限制collector的总数实现，但进度条显示原始帧，因为四帧压为一帧故原始帧为4倍，同时结尾会不到4帧，故最后进度条中的帧数会为3倍多
TOTAL_FRAMES = 5000000
#TOTAL_FRAMES = 10000
REPLAY_CAP=500_000 # 缓冲区的容量
#REPLAY_CAP=1000 # 缓冲区的容量
BUFFER_DIR="/home/larry/buffer/torchrl_rb" # 缓冲区如果全部存内存可能不够，一部分存硬盘，这里指定路径

# log
LOG_DIR = "./runs/breakout_dqn" # Tensorboard数据保存地点
CKPT_DIR = "./checkpoints" # checkpoints保存地点
LOAD_PATH = "./checkpoints/ckpt_13762.pt" # 恢复训练的地址
LOAD_CKPT = False# 是否恢复

# utils
## optim
LR=2.5E-4
ALPHA=0.95
EPS=0.01
MOMENTUM=0.0

# main
SEED=0 # 随机种子
N_STEP = 3
GAMMA=0.99 # 优化参数
BATCH=32
TRAIN_FREQ=4 # 多少帧叠成一帧，决定优化的次数
LEARN_STARTS=50_000 # 开始获取数据帧数
#LEARN_STARTS=1_000 # 开始获取数据帧数
TARGET_SYNC=10_000 # 多少帧同步网络
CKPT_EVERY_FRAMES = 100_000  # 保存权重的帧数

# policy
EPS_INIT=1.0 # 指定epsilon的变化
EPS_END=0.05
EPS_DECAY=2_000_000

