# 导入PyTorch库，用于构建深度学习模型。
import torch
# 导入NumPy库，用于数值计算。
import numpy as np
import torch.nn as nn
# 从scipy库中导入了linalg模块，并将其简写为la。SciPy是一个用于科学和技术计算的Python库，
# linalg子模块提供了线性代数功能，包括矩阵求逆、解线性方程组等。
import scipy.linalg as la

# setup_scatter可能是用来处理数据加载或预处理的函数；
# args可能包含了命令行参数或者全局配置设置；
# SoftmaxRegression则可能是实现了softmax回归分类器的类。
from dataloader import setup_scatter
from utils import args
from utils_classifier import SoftmaxRegression

# 特征提取器（FeatureExtractor）类和一些辅助函数的实现，用于深度学习模型中。
# 它使用了PyTorch库来构建神经网络，并用NumPy和SciPy来进行数值计算。
class FeatureExtractor(nn.Module):
    # 这是FeatureExtractor类的构造函数__init__。它接收一个params参数，默认值为args，这个参数可能是用来传递配置信息给类实例。
    # 构造函数首先调用了父类nn.Module的构造函数，然后将传入的参数保存到实例变量self.args中，并从中提取出与数据集相关的属性，
    # 比如数据集名称ds_name、数据集后缀ds_suffix和特征数量nr_fea。
    def __init__(self, params=args):
        super(FeatureExtractor, self).__init__()
        # 将传入的 params 参数保存到实例变量 self.args 中
        self.args = params
        # 从 self.args 中提取数据集名称 ds，并将其赋值给实例变量 self.ds_name。这可能是为了根据不同的数据集选择不同的特征提取器结构。
        self.ds_name = self.args.ds
        # 同样从 self.args 中提取数据集后缀 ds_suffix，并赋值给 self.ds_suffix。后缀可能用于进一步区分同一数据集的不同变体（例如，使用不同的预处理方法）。
        self.ds_suffix = self.args.ds_suffix
        # 提取特征数量 nr_fea，并赋值给 self.nr_fea。这可能是每个输入样本的特征维度。
        self.nr_fea = self.args.nr_fea
        # 检查数据集名称是否包含 "mnist"（不区分大小写）。如果是，则说明当前处理的是 MNIST 或 Fashion-MNIST 数据集。
        # self.feature_extractor_part1 = nn.Sequential(...)
        # 如果是 MNIST 数据集，定义 feature_extractor_part1，这是一个由多个卷积层和池化层组成的序列。具体来说：
        # nn.Conv2d(1, 20, kernel_size=5)：第一个卷积层，输入通道数为 1（灰度图像），输出通道数为 20，卷积核大小为 5x5。
        # nn.ReLU()：ReLU 激活函数，用于引入非线性。
        # nn.MaxPool2d(2, stride=2)：最大池化层，池化窗口大小为 2x2，步长为 2，用于下采样。
        # 第二个卷积层和池化层与第一个类似，但输入通道数为 20，输出通道数为 50。
        # [self.input_dim, self.H, self.L] = [50 * 4 * 4, 256, 128]
        # 设置一些中间变量，用于后续的全连接层。对于 MNIST 数据集，经过两次卷积和池化后，特征图的尺寸为 50x4x4，因此 input_dim 为 50 * 4 * 4 = 800。H 和 L 分别表示隐藏层和输出层的维度。
        if 'mnist' in self.ds_name.lower():  # for MNIST, FMNIST dataset
            self.feature_extractor_part1 = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )
            [self.input_dim, self.H, self.L] = [50 * 4 * 4, 256, 128]

        # 如果数据集名称包含 "sival"，则说明当前处理的是 SIVAL_MIPL 数据集。
        # 设置输入维度 input_dim 为 self.nr_fea，并定义两个隐藏层的维度 H1 和 H2，以及输出层的维度 L。这些值将用于后续的全连接层。
        elif 'sival' in self.ds_name.lower(): # for SIVAL_MIPL dataset
            [self.input_dim, self.H1, self.H2, self.L] = [self.nr_fea, 512, 1024, 256]

        # 如果数据集名称包含 "crc" 且后缀包含 "sift"，则说明当前处理的是 CRC-MIPL-sift 数据集。
        # 同样设置输入维度、隐藏层维度和输出层维度，但这次的维度与 SIVAL_MIPL 不同。
        elif 'crc' in self.ds_name.lower() and 'sift' in self.ds_name.lower(): # for CRC-MIPL-sift dataset
            [self.input_dim, self.H1, self.H2, self.L] = [self.nr_fea, 1024, 512, 64]

        # 如果数据集名称既不是 MNIST、SIVAL_MIPL 也不是 CRC-MIPL-sift，则假设是其他数据集（如 Birdsong_MIPL、CRC-MIPL 等）
        # 设置输入维度、隐藏层维度和输出层维度。
        else: # for Birdsong_MIPL, CRC-MIPL (row, sbn, kmeanSegs) dataset
            [self.input_dim, self.H, self.L] = [self.nr_fea, 512, 256]

        # 根据数据集名称和后缀，决定使用哪种类型的全连接层。如果数据集是 SIVAL_MIPL 或 CRC-MIPL-sift，则使用更复杂的全连接层结构（两层隐藏层），否则使用简单的全连接层结构（一层隐藏层）。
        # 定义 feature_extractor_part2，这是一个由多个全连接层、ReLU 激活函数和 Dropout 层组成的序列。Dropout 层用于防止过拟合。
        if 'sival' in self.ds_name.lower() or (
            'crc' in self.ds_name.lower() and 'sift' in self.ds_name.lower()):
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.input_dim, self.H1),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.H1, self.H2),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.H2, self.L),
                nn.ReLU(),
                nn.Dropout(),
            )
        else:
            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(self.input_dim, self.H),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(self.H, self.L),
                nn.ReLU(),
                nn.Dropout(),
            )

    # 前向传播方法 forward（其实就是输入x中做特征提取得到h）
    def forward(self, xs):
        # 将输入列表 xs[包数,实例数，特征维度] 中的所有张量沿着第 0 维度（通常是批次维度）连接起来，形成一个大的张量 x[包数*实例数，特征维度]-相当于包结构去掉并且把包内实例铺平。
        x = torch.cat(xs, dim=0)
        # 调用 setup_scatter 函数，该函数可能用于重新组织输入数据，并返回重新组织后的张量 x 以及一些索引信息 i 和指针 i_ptr，用于后续处理。
        # x：所有包中实例特征的拼接张量，形状为(N（包内的实例铺平）, 特征维度)。
        # i：每个实例所属的包索引，形状为(N,)。
        # i_ptr：每个包的起始位置，形状为(M+1,)，用于快速定位每个包的实例范围。
        x, i, i_ptr = setup_scatter(xs)
        # 使用 squeeze(0) 移除尺寸为 1 的维度，并将张量的数据类型转换为 32 位浮点数（float）。
        x = x.squeeze(0).float()
        # 如果数据集是 MNIST 或 Fashion-MNIST，则将输入张量 x 重塑为适合卷积操作的形状（批量大小, 通道数, 高度, 宽度），即 (batch_size, 1, 28, 28)。
        if 'mnist' in self.ds_name.lower():
            x = x.reshape(x.shape[0], 1, 28, 28)
            # 将重塑后的输入张量 x 通过 feature_extractor_part1，即卷积层部分，得到中间特征 h。
            h = self.feature_extractor_part1(x)
            # 将卷积层的输出 h 展平为一维向量，准备输入到全连接层中。展平后的维度为 50 * 4 * 4。
            h = h.view(-1, 50 * 4 * 4)
            # 将展平后的特征 h 通过 feature_extractor_part2，即全连接层部分，得到最终的特征表示。
            h = self.feature_extractor_part2(h)
        # 如果数据集不是 MNIST，则直接将输入张量 x 通过 feature_extractor_part2，跳过卷积层部分。
        else:   # for Birdsong_MIPL, SIVAL_MIPL, CRC-MIPL datasets
            x = x.float()
            h = self.feature_extractor_part2(x)
        # h 提取特征后的实例的隐向量，形状为(N（包内的实例铺平）, 特征维度)。
        # 创建一个空列表 hs，用于存储带包结构的实例。hs=[包数量,包内实例数、特征数量]
        hs = []
        # 遍历 i_ptr 中的起始和结束索引，i_ptr 是一个指针数组，用于指示每个包的边界。zip去掉第一和去掉最后一个并且把他打包，其实就是包内实例的坐标
        for start_idx, end_idx in zip(i_ptr[:-1], i_ptr[1:]):
            # 从 h 中提取每个包对应的特征，并将其添加到 hs 列表中。
            hs.append(h[start_idx:end_idx])
        # 返回包含hs，包含包结构的实例，恢复了包特征。 hs=[350-包数量,48-包内实例数、784-特征数量]
        return hs
    
# 2. 辅助函数 zb_regress 函数，定义了一个函数 zb_regress，它接收两个参数 x 和 f，并执行回归操作。
# 作用是从变量 x 中“回归掉”（regress out）由协变量 f 所解释的线性部分，从而得到去除 f 影响后的残差 x_out。
# 这是一种常见的协变量校正（covariate adjustment） 或 去混杂（confound removal） 技术，广泛应用于神经科学、基因组学、计量经济学和机器学习等领域。
# 假设你在做脑成像研究：
# x：每个受试者的全脑 fMRI 激活图（100 个体素 × 50 人）
# f：协变量矩阵，包含年龄、性别、头动参数等（50 人 × 3 列）
# 你想研究与认知任务相关的脑区激活，但不想让年龄等因素干扰结果。
# 调用 x_clean, _, _ = zb_regress(x.T, f)（注意转置以匹配样本维度），得到去除年龄/性别影响后的干净激活数据 x_clean，再进行后续分析。
def zb_regress(x, f):
    """
    regresses out f from x
    """
    # 1.计算 矩阵f 的伪逆（Moore-Penrose Pseudoinverse）f_inv。伪逆可以用于求解最小二乘问题。
    #   la 应该是 scipy.linalg 或 numpy.linalg 的别名。
    #   pinv(f) 计算矩阵 f 的伪逆，即使 f 不是方阵或不可逆也能求解。
    #   用途：用于求解最小二乘问题 min_b∥x−fb∥_2
    f_inv = la.pinv(f)
    # 2. 计算回归系数 b
    # 这等价于求解线性回归模型：x=fb+ε，x：目标变量（形状如 (n_samples, n_features)）；f：协变量/混杂因子（形状如 (n_samples, n_covariates)）；b：回归系数（形状如 (n_covariates, n_features)）
    # 使用伪逆 f_inv 对 x 进行回归，计算回归系数 b
    b = f_inv.dot(x)
    # 3. 计算残差（去除 f 影响后的 x）
    # 从 x 中减去 f 和 b 的乘积，得到调整后的特征 x_out。
    # f.dot(b) 是 f 对 x 的线性拟合值（预测值）。x_out 就是残差（residuals），即 x 中无法被 f 线性解释的部分。
    # 后续分析（如分类、聚类、相关性计算）可使用 x_out，以避免 f 的干扰。
    x_out = x - f.dot(b)
    # 将调整后的特征 x_out、回归系数 b 和伪逆矩阵 f_inv 组合成一个列表 out。
    out = [x_out, b, f_inv]
    # 返回包含 x_out、b 和 f_inv 的列表 out。
    # x_out：校正后的数据（主要输出）
    # b：回归系数（可用于解释 f 对 x 的影响强度）
    # f_inv：伪逆矩阵（较少用，可能用于后续计算）
    return out

# 函数 init_sr，用于初始化某些模型参数（变分推断或贝叶斯模型中的先验/后验参数）。
# 它接收四个参数 x、s、b 和 fiv，并用于初始化某些模型参数。
# 参数说明：
# x: 基于标准化后的包平均特征 nor_zb，形状通常为 [包数量, 特征数]
# s:包偏标签 s_array，形状 或 (包数量, n_classes)
# b: 回归系数 b（后续用于调整 alpha）
# fiv: 伪逆矩阵 fe_inv
def init_sr(x, s, b, fiv):
    # 创建一个 SoftmaxRegression 模型实例，设置学习率 eta、训练轮数 epochs、批量大小 minibatches、
    # 随机种子 random_seed、是否打印进度 print_progress 和类别数量 n_classes。
    model = SoftmaxRegression(
        eta=0.1,
        epochs=1000,
        minibatches=1,
        random_seed=args.seed,
        print_progress=0,
        n_classes=args.nr_class
    )

    # 训练softmax分类器，使用输入特征 x 和标签 s 训练 SoftmaxRegression 模型。
    model.fit(x, s)

    # 获取模型的偏置项 b_，并将其扩展为二维张量（增加一个维度）。
    alpha = model.b_[None]
    # 获取模型的权重 w_，即每个特征的系数。
    gamma = model.w_ 

    # Compute bag prediction u and reparametrize
    # 计算输入特征 x 与权重 gamma 的点积，得到预测值 u。
    u = x.dot(gamma)
    # 计算um=预测值 u 的均值  和 us= 预测值标准差 ，并将它们扩展为二维张量。
    [um, us] = [u.mean(0)[None], u.std(0)[None]]
    # 将偏置项 alpha 与预测值的均值 um 相加，进行重新参数化。
    alpha = alpha + um
    #  mu_gamma = 标准化后的权重 gamma，乘以预测值的标准差 us 并除以权重的平均平方根。
    mu_gamma = us * gamma / np.sqrt((gamma**2).mean(0)[None])
    # sd_gamma = mu_gamma 的标准差，乘以一个小常数 0.1 的平方根。
    sd_gamma = np.sqrt(0.1 * (mu_gamma**2).mean()) * np.ones_like(mu_gamma)
    # 更新偏置项 alpha，结合 fiv 和 b 进行调整。
    alpha = fiv.dot(np.ones((fiv.shape[1], 1))).dot(alpha) - b.dot(mu_gamma)

    # 初始化先验init prior
    # 计算 var_z = mu_gamma 和 sd_gamma 的 方差的均值。
    var_z = (mu_gamma**2 + sd_gamma**2).mean(0).reshape(1, -1)

    # 将 mu_gamma、sd_gamma、var_z 和 alpha 转换为 PyTorch 张量，并返回这些张量的列表。
    # mu_gamma = 标准化后的权重 gamma，乘以预测值的标准差 us 并除以权重的平均平方根。
    # sd_gamma = mu_gamma 的标准差，乘以一个小常数 0.1 的平方根。
    # var_z = mu_gamma 和 sd_gamma 的 方差的均值。
    # 偏置项 alpha，结合 fiv 和 b 进行调整。
    return [torch.Tensor(el) for el in (mu_gamma, sd_gamma, var_z, alpha)]

# generate_init_params:用于生成初始化参数，它接收三个参数 x、fe 和 s。
def generate_init_params(x, fe, s):
    # 定义一个小常数 eps，用于防止除零错误。
    eps = 1e-8
    # z_b=[包数量，特征数],表示包特征（平均值），并将其沿第一个维度拼接成一个大矩阵。
    z_b = np.concatenate([x.mean(0, keepdims=True) for x in x], axis=0) # generating features of each bag
    # 将 fe 和 s 转换为 NumPy 数组，方便后续的数值计算。
    [fe_array, s_array] = [fe.numpy(), s.numpy()]

    # 调用 zb_regress 函数，从 z_b 中回归出 fe_array 的影响，得到调整后的特征 zb、回归系数 b 和伪逆矩阵 fe_inv。
    zb, b, fe_inv = zb_regress(z_b, fe_array)
    # nor_zb = 对 zb 进行标准化，减去均值并除以标准差，确保特征的尺度一致。
    nor_zb = (zb - zb.mean(0, keepdims=True)) / (zb.std(0, keepdims=True) * np.sqrt(zb.shape[-1]) + eps)

    # 调用 init_sr 函数，入参：基于标准化后的包平均特征 nor_zb、包偏标签 s_array、回归系数 b 和伪逆矩阵 fe_inv，
    # 返回结果：生成初始化参数 mu_z、sd_z、var_z 和 alpha。
    mu_z, sd_z, var_z, alpha = init_sr(nor_zb, s_array, b, fe_inv)
    # 返回生成的初始化参数。
    # mu_gamma = 标准化后的权重 gamma，乘以预测值的标准差 us 并除以权重的平均平方根。
    # sd_gamma = mu_gamma 的标准差，乘以一个小常数 0.1 的平方根。
    # var_z = mu_gamma 和 sd_gamma 的 方差的均值。
    # 偏置项 alpha，结合 fiv 和 b 进行调整。
    return mu_z, sd_z, var_z, alpha
