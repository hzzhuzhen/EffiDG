import numpy as np
import torch
# 从 torch.distributions 模块中导入 LowRankMultivariateNormal 类。这个类用于表示低秩多变量正态分布，它可以高效地处理高维数据，尤其适用于贝叶斯推断中的后验分布近似。
from torch.distributions import LowRankMultivariateNormal

# BayesianPosterior 的类，该类继承自 torch.nn.Module，用于表示贝叶斯后验分布。
# 它使用了 LowRankMultivariateNormal 分布来近似复杂的多变量正态分布，并提供了从该分布中采样的方法。
# 这段代码实现了一个贝叶斯后验分布的近似，使用 LowRankMultivariateNormal 来表示分布，并提供了从该分布中采样的方法。BayesianPosterior 类的主要功能包括：
# 初始化均值和协方差矩阵的参数。
# 构建低秩多变量正态分布。
# 提供从分布中采样或获取均值的方法，支持训练和测试两种模式。
# 这种设计允许模型在训练过程中通过采样来捕捉不确定性，而在测试时可以使用均值来进行预测，从而结合了贝叶斯推断的优势。
class BayesianPosterior(torch.nn.Module):
    # 定义类的构造函数 __init__，它接收三个参数：
    # nr_xs：输入特征的数量。
    # nr_out：输出的数量（通常是类别数或回归目标数）。
    # params：一个包含初始化参数的列表或元组，通常包括均值 mu_z 和标准差 sd_z 等。
    def __init__(self, nr_xs, nr_out, params):
        super().__init__()
        # 定义一个小常数 eps，用于防止数值计算中的除零错误或其他不稳定情况。
        eps = 1e-4
        # 将输入特征的数量 nr_xs 保存到实例变量 self.nr_input 中，以便在类的其他方法中使用。
        self.nr_input = nr_xs
        # 将输出的数量 nr_out 保存到实例变量 self.nr_out 中。
        self.nr_out = nr_out

        # 从 params 中解包出均值 mu_z 和标准差 sd_z，并忽略其他参数（使用 _ 表示忽略）。mu_z 和 sd_z 可能是传入的初始化参数，分别表示某些特征的均值和标准差。
        mu_z, sd_z, *_ = params
        # 将 mu_z 和 sd_z 转换为 PyTorch 张量，并对它们进行转置（T），以确保它们的形状符合后续操作的要求。
        [mu_z, sd_z] = [torch.Tensor(mu_z.T), torch.Tensor(sd_z.T)]
        # 初始化 mu_u 和 sd_u，分别表示另一部分特征的均值和标准差。mu_u 被初始化为全零张量，而 sd_u 被初始化为小常数 sqrt(0.1) 的张量。这可能是为了给某些特征（如噪声项）赋予较小的初始方差。
        [mu_u, sd_u] = [torch.zeros_like(mu_z), torch.sqrt(0.1 * torch.ones_like(sd_z))]
        # 将 mu_u 和 mu_z 沿着第二个维度（特征维度）拼接起来，形成完整的均值向量 mu。同样地，将 sd_u 和 sd_z 拼接起来，形成完整的标准差向量 sd。mu 和 sd 分别表示整个分布的均值和标准差。
        [mu, sd] = [torch.cat([mu_u, mu_z], 1), torch.cat([sd_u, sd_z], 1)]

        # 使用 torch.diag_embed 函数将标准差向量 sd 对角化，生成一个对角矩阵 sd_diag。这个矩阵的对角线元素是 sd，其余元素为零。对角矩阵用于表示协方差矩阵的对角部分。
        sd_diag = torch.diag_embed(sd)
        # [cov_factor, cov_ldiag] = [...]
        # 初始化两个参数 cov_factor 和 cov_ldiag，它们用于构建低秩多变量正态分布的协方差矩阵：
        # cov_factor：通过对 sd_diag 添加一个小扰动（eps * torch.randn(sd_diag.shape)）来初始化。这个扰动使得协方差矩阵不再是严格的对角矩阵，而是具有一定的低秩结构。cov_factor 用于表示协方差矩阵的低秩部分。
        # cov_ldiag：初始化为 log(eps) 的张量，表示协方差矩阵对角部分的对数标准差。cov_ldiag 用于控制协方差矩阵的对角元素。
        [cov_factor, cov_ldiag] = [
            eps * torch.randn(sd_diag.shape) + sd_diag, 
            np.log(eps) * torch.ones(nr_out, self.nr_input * 2)
        ]

        # 将均值向量 mu 注册为模型的参数（torch.nn.Parameter），这意味着它将在训练过程中被优化。mu 表示分布的均值。
        self.mu = torch.nn.Parameter(mu)
        # 将 cov_factor 注册为模型的参数，表示协方差矩阵的低秩部分。cov_factor 用于构建协方差矩阵的非对角部分。
        self.cov_factor = torch.nn.Parameter(cov_factor)
        # 使用 register_buffer 方法将 cov_ldiag 注册为缓冲区（buffer），而不是参数。缓冲区不会参与梯度计算，但会在模型保存和加载时一起保存。cov_ldiag 用于表示协方差矩阵对角部分的对数标准差。
        self.register_buffer("cov_ldiag", cov_ldiag)

    # 3. 定义 distribution 属性
    # 使用 @property 装饰器将 distribution 方法定义为属性，这样可以在类外部像访问普通属性一样访问它，而不需要调用方法。
    # 定义 distribution 方法，返回一个 LowRankMultivariateNormal 实例，表示当前的贝叶斯后验分布。
    @property
    def distribution(self):
        # 创建一个 LowRankMultivariateNormal 实例 dist，使用以下参数：
        # self.mu：分布的均值向量。
        # self.cov_factor：协方差矩阵的低秩部分。
        # torch.exp(self.cov_ldiag)：协方差矩阵对角部分的标准差（通过对 cov_ldiag 进行指数运算得到）。
        dist = LowRankMultivariateNormal(
            self.mu, self.cov_factor, torch.exp(self.cov_ldiag))
        # 返回构建好的 LowRankMultivariateNormal 分布对象。
        return dist
    
    # 4. 定义 get_beta 方法【获取beta_u和beta_z核心方法】
    # 定义 get_beta 方法，用于从贝叶斯后验分布中采样或获取均值。该方法接收两个参数：
    # n_samples：采样的样本数量。
    # is_test：布尔值，指示是否处于测试模式。如果为 True，则直接返回均值；否则进行采样。应该随机，每次采样不同以反映不确定性
    # beta_u 是后n_samples个采样是随机特征，用于建模 实例级不确定性（如噪声、混杂因子），希望其稳定，代表“真实”信号，beta_u 的后验方差较大，采样结果波动明显 → 体现“随机性”。beta_u 被当作辅助噪声项或校正项，只在训练时引入随机性以提升鲁棒性，测试时虽用均值但重要性较低。
    # beta_z 是前n_samples个采样作为固定特征。用于建模 包级/全局判别特征（如类别原型、共享权重），beta_z 的后验方差可能被约束得很小（例如通过先验或正则化），导致其采样结果几乎不变 → 近似“固定。beta_z被当作模型的核心可学习参数（类似传统神经网络权重），用于最终分类；
    def get_beta(self, n_samples, is_test):
        # 如果处于测试模式（is_test == True），则直接返回均值 mu 的转置版本 mu_T，并从中提取两部分：
        # beta_u：表示 mu 的前 self.nr_input 个特征，对应于 u 部分的均值。
        # beta_z：表示 mu 的后 self.nr_input 个特征，对应于 z 部分的均值。
        if is_test:
            # 将均值向量 self.mu 进行转置，得到 mu_T，方便后续的切片操作。
            mu_T = self.mu.T
            # beta_u = 从 mu_T 中提取前 self.nr_input 个特征，表示 u 部分的均值，并使用 unsqueeze(2) 增加一个维度，使其形状适合后续操作。
            beta_u = mu_T[:self.nr_input].unsqueeze(2)
            # beta_z = 从 mu_T 中提取后 self.nr_input 个特征，表示 z 部分的均值，并同样增加一个维度。
            beta_z = mu_T[self.nr_input:].unsqueeze(2)
        # 如果不处于测试模式（is_test == False），则从分布中进行采样：
        else:
            # 使用 rsample 方法从 LowRankMultivariateNormal 分布中采样 n_samples 个样本。rsample 支持自动求导，因此可以在训练过程中使用。
            sampling_results = self.distribution.rsample([n_samples])
            # 对采样结果进行维度重排，使样本的第一个维度为特征维度，第二个维度为输出维度，第三个维度为样本数量。
            reshaped_samples = sampling_results.permute([2, 1, 0])
            # 从重排后的采样结果中提取后 self.nr_input 个样本[128个]，表示 u 部分的采样结果，比较随机的部分认为是噪音或者包特性的属性，做包内特征。。
            beta_u = reshaped_samples[:self.nr_input, :, :]
            # 从重排后的采样结果中提取前 self.nr_input 个样本，表示 z 部分的采样结果，比较稳定的部分每次都差不多，做为挎包通用特征。
            beta_z = reshaped_samples[self.nr_input:, :, :]
        # 返回 beta_u 和 beta_z，分别表示 u 和 z 部分的均值或采样结果。
        return beta_u, beta_z


