# 1. 导入必要的库
# torch：PyTorch 库，用于构建和训练神经网络。
# nn：PyTorch 中的神经网络模块，提供了常用的层和损失函数。
# copy：用于深拷贝对象，确保每个复制的模块是独立的。
# Variable：在较早版本的 PyTorch 中用于表示需要计算梯度的张量。在 PyTorch 0.4.0 及以上版本中，Variable 已经被弃用，直接使用 torch.Tensor 即可。
from torch import nn
import torch
import copy
from torch.autograd import Variable

# 2. 辅助函数：clones
# 功能：这个函数用于创建 N 个相同的神经网络层。它接收一个 module（即一个神经网络层或模块），
# 并返回一个包含 N 个该模块的 nn.ModuleList。
# 用途：通常用于构建具有相同结构的多层神经网络，例如 Transformer 模型中的多头自注意力机制。
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 3. 神经网络模型：DeepNet
# 多个神经网络模型类，用于多标签分类任务。
# 这些模型使用 PyTorch 框架构建，
# 并且都包含了前向传播（forward）和获取特征（forward_and_get_fea）的功能
class DeepNet(nn.Module):
    # __init__：
    # dim_x：输入特征的维度。
    # dim_y：输出标签的维度（即标签数量）。
    # hidden：隐藏层的神经元数量列表，默认为 [64, 64]。hidden[0] 和 hidden[1] 分别表示第一层和第二层隐藏层的神经元数量。
    # self.net：一个包含两层全连接层和 ReLU 激活函数的序列模块，用于提取特征。
    # self.fc：最后一层全连接层，将特征映射到输出标签空间。
    # nn.functional.sigmoid(y)：使用 Sigmoid 激活函数将输出转换为概率值，适用于多标签分类任务。
    def __init__(self, dim_x, dim_y, hidden=None):
        super(DeepNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_x, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden[1], dim_y)
        )
        # Sigmoid is necessary to get smooth score

    # forward：
    # 功能：前向传播函数，接收输入 x，经过特征提取层 self.net
    # 和输出层 self.fc，最后通过 Sigmoid 函数得到预测结果 y。
    # 返回值：返回预测的标签概率 y。
    def forward(self, x):
        fea = self.net(x)
        y = self.fc(fea)

        y = nn.functional.sigmoid(y)
        
        return y

    # forward_and_get_fea：
    # 功能：与 forward 类似，但同时返回中间特征 fea 和最终的预测结果 y。
    # 这在某些情况下非常有用，例如需要分析模型的内部表示或进行特征可视化。
    def forward_and_get_fea(self, x):
        fea = self.net(x)
        y = self.fc(fea)

        y = nn.functional.sigmoid(y)
        
        return y, fea

# 4. 神经网络模型：DeepNet_Large
# __init__：
# dim_x：输入特征的维度。
# dim_y：输出标签的维度。
# hidden：隐藏层的神经元数量列表，固定为 [1024, 512, 256, 64]。这是一个更深、更复杂的网络结构，适合处理更大规模的数据集或更复杂的任务。
# self.net：一个包含四层全连接层和 ReLU 激活函数的序列模块，用于提取特征。
# self.fc：最后一层全连接层，将特征映射到输出标签空间。
# nn.functional.sigmoid(y)：使用 Sigmoid 激活函数将输出转换为概率值。
# forward 和 forward_and_get_fea：
# 与 DeepNet 类似，分别用于前向传播和同时返回特征与预测结果。
class DeepNet_Large(nn.Module):
    # __init__：
    # dim_x：输入特征的维度。
    # dim_y：输出标签的维度。
    # hidden：隐藏层的神经元数量列表，固定为 [1024, 512, 256, 64]。这是一个更深、更复杂的网络结构，适合处理更大规模的数据集或更复杂的任务。
    # self.net：一个包含四层全连接层和 ReLU 激活函数的序列模块，用于提取特征。
    # self.fc：最后一层全连接层，将特征映射到输出标签空间。
    # nn.functional.sigmoid(y)：使用 Sigmoid 激活函数将输出转换为概率值。
    def __init__(self, dim_x, dim_y):
        super(DeepNet_Large, self).__init__()

        hidden = [1024, 512, 256, 64]

        self.net = nn.Sequential(
            nn.Linear(dim_x, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], hidden[2]),
            nn.ReLU(),
            nn.Linear(hidden[2], hidden[3]),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden[3], dim_y)
        )
        # Sigmoid is necessary to get smooth score

    # forward 和 forward_and_get_fea：
    # 与 DeepNet 类似，分别用于前向传播和同时返回特征与预测结果。
    def forward(self, x):
        fea = self.net(x)
        y = self.fc(fea)

        y = nn.functional.sigmoid(y)
        
        return y

    def forward_and_get_fea(self, x):
        fea = self.net(x)
        y = self.fc(fea)

        y = nn.functional.sigmoid(y)
        
        return y, fea
# 5. 神经网络模型：LinearClassifier
class LinearClassifier(nn.Module):
    # __init__：
    # dim_x：输入特征的维度。
    # dim_y：输出标签的维度。
    # self.fc：一个简单的线性层，直接将输入特征映射到输出标签空间，没有隐藏层。
    # nn.functional.sigmoid(y)：使用 Sigmoid 激活函数将输出转换为概率值。
    def __init__(self, dim_x, dim_y):
        super(LinearClassifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(dim_x, dim_y)
        )
        # Sigmoid is necessary to get smooth score

    # forward：
    # 功能：前向传播函数，接收输入 x，经过线性层 self.fc，最后通过 Sigmoid 函数得到预测结果 y。
    # 返回值：返回预测的标签概率 y。
    def forward(self, x):
        y = self.fc(x)

        y = nn.functional.sigmoid(y)
        
        return y
