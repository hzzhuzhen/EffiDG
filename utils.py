#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# import os
# 导入 os 模块，用于与操作系统交互，例如创建目录、检查文件是否存在等。
# import sys
# 导入 sys 模块，用于访问与 Python 解释器相关的功能，例如标准输入输出流。
# import time
# 导入 time 模块，用于获取当前时间，通常用于日志文件命名或记录训练时间。
# import logging
# 导入 logging 模块，用于记录日志信息，包括调试信息、警告、错误等。
# import numpy as np
# 导入 NumPy 库，并将其简写为 np，用于科学计算和数组操作。
# import torch
# 导入 PyTorch 库，用于深度学习模型的构建和训练。
# import random
# 导入 random 模块，用于生成随机数，常用于设置随机种子以保证实验的可重复性。
# import argparse
# 导入 argparse 模块，用于解析命令行参数，方便用户通过命令行传递配置信息。
import os
import sys
import time
import logging
import numpy as np
import torch
import random
import argparse
 
# 3. 定义 str2bool 函数
# def str2bool(v):
# 定义了一个函数 str2bool，用于将字符串转换为布尔值。这个函数主要用于处理命令行参数中的布尔类型参数。
# if isinstance(v, bool)
# 如果传入的值已经是布尔类型，则直接返回该值。
# if v.lower() in ('yes', 'true', 't', 'y', '1'):
# 如果传入的字符串是 'yes'、'true'、't'、'y' 或 '1'（不区分大小写），则返回 True。
# elif v.lower() in ('no', 'false', 'f', 'n', '0'):
# 如果传入的字符串是 'no'、'false'、'f'、'n' 或 '0'（不区分大小写），则返回 False。
# else:
# 如果传入的字符串既不是表示 True 的值也不是表示 False 的值，则抛出 argparse.ArgumentTypeError 异常，提示用户输入无效的布尔值。
def str2bool(v): 
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# training settings
# 4. 设置命令行参数
# parser = argparse.ArgumentParser(...)
# 创建一个 ArgumentParser 对象 parser，用于解析命令行参数。description 参数提供了脚本的简短描述，帮助用户理解脚本的功能。
# parser.add_argument(...)
# 使用 add_argument 方法为 parser 添加多个命令行参数。每个参数都有以下属性：
# --no-cuda：布尔参数，用于禁用 CUDA 训练。默认值为 False，即默认启用 CUDA。
# --epochs：整数参数，表示训练的轮数，默认值为 100。
# --reg：浮点数参数，表示权重衰减（L2 正则化），默认值为 10e-5。
# --seed：整数参数，表示随机种子，默认值为 123，用于保证实验的可重复性。
# --data_path：字符串参数，表示数据集路径，默认值为 ./data。
# --exp_dir：字符串参数，表示日志文件保存路径，默认值为 ./logs/。
# --index：字符串参数，表示索引文件路径，默认值为 index。
# --ds：字符串参数，表示数据集名称，默认值为 MNIST_MIPL，支持不同的数据集。
# --ds_suffix：字符串参数，表示数据集的具体类型，默认值为 1。
# --bs：整数参数，表示批量大小，默认值为 1。
# --nr_fea：整数参数，表示每个实例的特征维度，默认值为 784（MNIST 图像的像素数）。
# --nr_class：整数参数，表示类别数量，默认值为 5。
# --nr_samples：整数参数，表示采样次数，默认值为 8。
# --nr_trial：整数参数，表示试验次数，默认值为 1。
# --normalize：布尔参数，表示是否对数据集进行归一化，默认值为 False。
# --lr：浮点数参数，表示学习率，默认值为 0.0005。
# --smoke_test：布尔参数，表示是否进行烟雾测试（快速测试），默认值为 False。
# --debug：布尔参数，表示是否启用调试模式，默认值为 False。
parser = argparse.ArgumentParser(
    description='Fast Multi-Instance Partial-Label Learning')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R', help='weight decay')
parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed (default: 123)')
parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
parser.add_argument('--exp_dir', type=str, default='./logs_v3/', help='logfile path')
parser.add_argument('--index', type=str, default='index', help='index path')
parser.add_argument('--ds', type=str, default='MNIST_MIPL', help='MNIST_MIPL, FMNIST_MIPL, ...')
parser.add_argument('--ds_suffix', type=str, default= None, help='the specific type of the data set')
parser.add_argument('--bs', type=int, default=1, help='batch size for training')
parser.add_argument('--nr_fea', type=int, default=784, help='feature dimension of an instance')
parser.add_argument('--nr_class', type=int, default=5, help='classes of bag')
parser.add_argument('--nr_samples', type=int, default=8, help='number of sampling')
parser.add_argument('--nr_trial', type=int, default=1, help='number of trial')
parser.add_argument('--normalize', type=str2bool, default=False, help='normalize the dataset, True or False')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate (default: 0.0005)')
parser.add_argument('--smoke_test', type=str2bool, default=False, help='smoke_test, True or False')
parser.add_argument('--debug', type=str2bool, default=False, metavar='Debug Flag', help='debug flagm True or False')
# 流行学习的参数

# 0.2流行学习方法加进来的
# --epochs：总训练轮数，默认为 300。
# --batch-size：训练时的批量大小，默认为 32。
# --lr 或 --learning-rate：初始学习率，默认为 0.01。
# --momentum：SGD 优化器的动量，默认为 0.9。
# --weight_decay：权重衰减（L2 正则化）系数，默认为 5e-5。
# --neighbors_num：构建图时使用的 k 近邻数，默认为 10。
# --threshold：评估时的阈值，默认为 0.7。
# --hidden_size：隐藏层的神经元数量，默认为 '64,64'，表示两个隐藏层，每层 64 个神经元。
# --gpuid：使用的 GPU ID，默认为 0。
parser.add_argument('--using_lp', action='store_true')
parser.add_argument('--using_prep', action='store_true')
parser.add_argument('--class_cons', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--threshold', type=float, default=0.7)
parser.add_argument('--hidden_size', type=str, default='64,64')
parser.add_argument('--gpuid', type=int, default=0)
# main parameters
# --alpha：标签传播中基于图结构传播项的权重，默认为 0.01。
# --eta：标签传播中基于初始标签分布约束项的权重，默认为 1。
# --beta：标签传播中基于标签相关性约束项的权重，默认为 0.01。
# --maxiter：标签传播的最大迭代次数，默认为 200。太多了，按我的逻辑只能搞1次即可
# --gamma：标签传播的学习率，默认为 0.01。
# --tr_rate：训练集与测试集的比例，默认为 0.9。
# --no-verbose：是否禁用日志输出，默认为 False。
# --mode：模式选择，默认为 0。
# --eval_every：每隔多少个 epoch 进行一次评估，默认为 5。
# --using_lp：是否使用标签传播算法，默认为 False。
# --using_prep：是否使用预处理数据，默认为 False。
parser.add_argument('--tr_rate', type=float, default=0.9)
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--eval_every', type=int, default=5)
parser.add_argument('--gamma', type=float, default=10., metavar='L', help='the weight of the inhibition loss 抑制损失权重')
parser.add_argument('--maxiter', type=int, default=1)
parser.add_argument('--neighbors_num', type=int, default=10)
parser.add_argument('--no-verbose', action='store_true')
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--eta', type=float, default=1)
parser.add_argument('--beta', type=float, default=0.00)
parser.add_argument('--lambda_recon_rate', type=float, default=0.1, metavar='L', help='manifold在更新参数权重比例，默认10%')
#parser.add_argument('--mu', type=float, default=0.2, metavar='L', help='the weight of the sparsity loss 稀疏损失权重')
#parser.add_argument('--mu2', type=float, default=0.05, metavar='L', help='the weight of the sparsity loss 稀疏损失权重')
parser.add_argument('--L', type=int, default=500, metavar='S', help='the embedded space dimension 嵌入空间权重')
#parser.add_argument('--la', type=float, default=2.0, metavar='L', help='残差 max的权重')
#parser.add_argument('--mix_loss_rate', type=float, default=0.1, metavar='L', help='混合损失的权重')
#parser.add_argument('--isMixLoss', type=str2bool, default=False, help='是否开启混合注意力损失，True 或 False')
parser.add_argument('--current_epoch', type=int, default=0, metavar='S', help='当前epoch的轮次')
parser.add_argument('--second_lambda_recon_rate', type=float, default=0.0001, metavar='L', help='Topk实例平均特征下的流行学习Y_recon）')
parser.add_argument('--lambda_recon_X_rate', type=float, default=0.0001, metavar='L', help='X_recon比例')
parser.add_argument('--lambda_recon_X_topk_rate', type=float, default=0.0001, metavar='L', help='X_recon_topk比例')






args = parser.parse_args()
# 5. 配置设备（CPU 或 GPU）
# out_dir = args.exp_dir
# 将日志文件保存路径 args.exp_dir 赋值给变量 out_dir，以便后续使用。
# if not args.no_cuda and torch.cuda.is_available():
# 检查是否启用了 CUDA 并且系统上是否有可用的 GPU。如果满足条件，则将设备设置为 GPU；否则，将设备设置为 CPU。
# device = torch.device("cuda")
# 如果使用 GPU，则创建一个 torch.device("cuda") 对象，表示使用 GPU 进行计算。
# args.cuda = True
# 将 args.cuda 设置为 True，表示使用 GPU。
# else:
# 如果不使用 GPU，则创建一个 torch.device("cpu") 对象，表示使用 CPU 进行计算。
# args.cuda = False
# 将 args.cuda 设置为 False，表示使用 CPU。
out_dir = args.exp_dir
if not args.no_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    args.cuda = True
else:
    device = torch.device("cpu")
    args.cuda = False

# 6. 日志
# if not os.path.exists(out_dir):
# 检查日志文件夹 out_dir 是否存在。如果不存在，则创建该文件夹。
# os.makedirs(out_dir)
# 使用 os.makedirs 创建日志文件夹。如果文件夹已经存在，则不会报错。
# LOG_FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() ]%(levelname)s >>> %(message)s"
# 定义日志格式字符串 LOG_FORMAT，用于控制日志的输出格式。格式中包含时间戳、文件名、行号、函数名、日志级别和日志消息。
# log_name = "{}_{}_{}_lr{}epoch{}nrS{}.log".format(...)
# 生成日志文件的名称，文件名中包含了数据集名称、数据集后缀、当前时间、学习率、训练轮数和采样次数。这样可以确保每次运行时生成唯一的日志文件。
# if args.debug:
# 如果启用了调试模式（args.debug == True），则将日志输出到标准输出流 sys.stdout，并且日志级别设置为 DEBUG，记录所有调试信息。
# else:
# 如果未启用调试模式，则将日志输出到文件，文件路径为 os.path.join(out_dir, log_name)，并且日志级别设置为 INFO，只记录重要的信息。
# logging.info("Record the params {}\n".format(args))
# 记录当前的参数配置信息到日志文件中，方便后续查看和调试。
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Declare the log file
LOG_FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)20s() ]%(levelname)s >>> %(message)s"

log_name = "{}_{}_{}_lr{}epoch{}nrS{}.log".format(
    args.ds, args.ds_suffix, 
    time.strftime("%Y%m%d_%H%M%S", time.localtime()),
    "".join(str(args.lr).split(".")), args.epochs, args.nr_samples
    )

if args.debug:
    logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FORMAT,
        stream=sys.stdout
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(out_dir, log_name),
        filemode="w",
        format=LOG_FORMAT
    )

logging.info("Record the params {}\n".format(args))

# 8. 设置随机种子
# def seed_everything(seed):
# 定义一个函数 seed_everything，用于设置全局的随机种子，确保实验的可重复性。
# random.seed(seed)
# 设置 Python 内置的 random 模块的随机种子，确保生成的随机数序列相同。
# np.random.seed(seed)
# 设置 NumPy 的随机种子，确保 NumPy 生成的随机数序列相同。
# torch.manual_seed(seed)
# 设置 PyTorch 的 CPU 随机种子，确保 PyTorch 在 CPU 上生成的随机数序列相同。
# torch.cuda.manual_seed(seed)
# 设置 PyTorch 的当前 GPU 随机种子，确保 PyTorch 在当前 GPU 上生成的随机数序列相同。
# torch.cuda.manual_seed_all(seed)
# 设置 PyTorch 的所有 GPU 随机种子，确保 PyTorch 在所有 GPU 上生成的随机数序列相同（适用于多 GPU 环境）。
# torch.backends.cudnn.benchmark = False
# 关闭 CuDNN 的自动优化功能，确保每次卷积操作的行为一致，避免由于 CuDNN 优化导致的结果不一致。
# torch.backends.cudnn.deterministic = True
# 启用 CuDNN 的确定性模式，确保卷积操作的结果完全一致。
# os.environ['PYTHONHASHSEED'] = str(seed)
# 设置 Python 的哈希种子，确保 Python 的哈希函数行为一致。
# torch.use_deterministic_algorithms(True)
# 启用 PyTorch 的确定性算法，确保所有操作的行为一致。
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
# 设置 CuBLAS 的工作区配置，确保 CuBLAS 的行为一致。:16:8 是一种常见的配置，用于平衡性能和确定性。
def seed_everything(seed):
    random.seed(seed)                                       # Python random module
    np.random.seed(seed)                                    # Numpy module
    torch.manual_seed(seed)                                 # Current CPU
    torch.cuda.manual_seed(seed)                            # Current GPU
    torch.cuda.manual_seed_all(seed)                        # All GPU (Optional)
    torch.backends.cudnn.benchmark = False                  # Close optimization
    torch.backends.cudnn.deterministic = True               # Close optimization
    os.environ['PYTHONHASHSEED'] = str(seed)                # Python hash seed
    torch.use_deterministic_algorithms(True)                # Enable deterministic algorithms
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'         # Set CUBLAS workspace

