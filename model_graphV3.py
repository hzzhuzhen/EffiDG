import os  # 提供与操作系统交互的功能
import sys
# 获取当前文件的路径
current_path = os.path.dirname(os.path.abspath(__file__))
# 获取上级文件的路径
parent_path = os.path.dirname(current_path)
# 获取上上级目录的路径
grandparent_path = os.path.dirname(os.path.dirname(current_path))
# 将上上级目录添加到sys.path
sys.path.insert(0, grandparent_path)
sys.path.insert(1, parent_path)
sys.path.insert(2, current_path)
print('当前第一优先级目录:', sys.path[0])
# 将上上级目录添加到sys.path
import torch
import numpy as np
# 从PyTorch中导入优化器模块，如SGD、Adam等。
import torch.optim as optim

# 从PyTorch的分布模块中导入KL散度计算函数，用于衡量两个概率分布之间的差异
from torch.distributions.kl import kl_divergence
# 从torch_scatter库中导入scatter_softmax和segment_add_csr函数，用于高效地处理稀疏矩阵操作。
from torch_scatter import scatter_softmax, segment_add_csr
# 从PyTorch的分布模块中导入LowRankMultivariateNormal类，用于表示低秩多元正态分布
from torch.distributions import LowRankMultivariateNormal

# 从utils模块中导入日志记录工具logging和设备配置device，用于控制输出和指定计算设备（CPU或GPU）。
from utils import logging, device
# 从dataloader模块中导入setup_scatter函数，用于设置数据加载时的分散操作
from dataloader import setup_scatter
# 从utils_nn模块中导入generate_init_params函数和FeatureExtractor类，分别用于生成初始化参数和特征提取器。
from utils_nn import generate_init_params
from utils_nn import FeatureExtractor
# 从utils_posterior模块中导入BayesianPosterior类，用于贝叶斯后验推断(现有观测数据再后验推断参数值)。
from utils_posterior import BayesianPosterior
from graph.deep_manifold_gpu_v5 import *

# 名为EffiDG的PyTorch模块，用于实现多实例学习（MIL, Multiple Instance Learning）中的贝叶斯推理，
# 基于多实例学习的贝叶斯模型，结合了特征提取、后验分布估计和损失函数计算等功能
class EffiDG(torch.nn.Module):
    # 定义构造函数，接收两个参数：cfg（配置参数）和params（初始化参数，默认为[None, 1, 1]）。
    def __init__(self, cfg, params=[None, 1, 1]):
        super().__init__()
        self.args = cfg
        self.predict_flag = False
        # 配置参数中获取类别数量
        self.nr_class = self.args.nr_class
        # 从配置参数中获取样本数量
        self.nr_samples = self.args.nr_samples
        # 创建一个FeatureExtractor实例，用于提取特征。
        self.fea_extractor = FeatureExtractor(self.args)
        # 解析参数params，分别赋值给init_params(默认none)、self.dim_xs(默认1)和nr_fixed_effects(默认1)。
        [init_params, self.dim_xs, nr_fixed_effects] = params
        # 初始化ln_sigma_u=[1,类别数]表示不确定性的方差为 0.5 的高斯分布的标准差的自然对数。
        # 表示先验分布的标准差对数，初始值为0.5 * log(0.5)。初始假设每个类别的“不确定性”对应一个方差为 0.5 的高斯分布（≈ -0.3466）。
        ln_sigma_u = torch.full((1, self.nr_class), 0.5 * np.log(0.5))
        if init_params is not None:
            # 解包init_params，忽略前面的部分，提取var_z和alpha。
            *_, var_z, alpha = init_params
            # ln_sigma_z=表示后验分布的标准差对数，根据var_z计算出
            ln_sigma_z = 0.5 * torch.log(var_z)
        else:
            # 初始化alphap[1,类别数]为全零矩阵，表示固定效应参数。
            alpha = torch.zeros((nr_fixed_effects, self.nr_class))
            # 初始化ln_sigma_z[1,类别数]，表示后验分布的标准差对数，初始值为0.5 * log(0.5)。也是方差为 0.5 的高斯分布的标准差的自然对数。
            ln_sigma_z = torch.full((1, self.nr_class), 0.5 * np.log(0.5))

        # 将alpha、ln_sigma_u、ln_sigma_z和posterior注册为模型参数，
        # 其中posterior是一个BayesianPosterior实例，用于贝叶斯后验推断，其中默认参数1，标签类别数，先验ln_sigma_u+后验ln_sigma_z。
        [self.alpha, self.ln_sigma_u, self.ln_sigma_z, self.posterior] = [
            torch.nn.Parameter(alpha), 
            torch.nn.Parameter(ln_sigma_u), 
            torch.nn.Parameter(ln_sigma_z), 
            BayesianPosterior(self.dim_xs, self.nr_class, init_params)
        ]


    # 2.2 EffiDG模型初始化
    # 定义一个静态方法initialize_model，用于初始化模型。
    def initialize_model(x, fe, s, cfg_params):
        # 创建一个特征提取器实例（初始化的参数传入）。
        fea_extractor = FeatureExtractor(cfg_params)
        # 使用特征提取器处理输入数据x，得到特征表示xs。
        xs = fea_extractor(x)
        # 将特征表示转换为NumPy数组，便于后续处理。
        xs_array = [i_x.detach().numpy() for i_x in xs]
        # 生成初始参数init_params，包括权重alpha、方差var_z等。
        # mu_gamma = 标准化后的权重 gamma，乘以预测值的标准差 us 并除以权重的平均平方根。
        # sd_gamma = mu_gamma 的标准差，乘以一个小常数 0.1 的平方根。
        # var_z = mu_gamma 和 sd_gamma 的 方差的均值。
        # 偏置项 alpha，结合 fiv 和 b 进行调整。
        init_params = generate_init_params(xs_array, fe, s)
        # 构造模型参数列表，包含初始参数、特征维度和固定效应数量。
        model_params = [init_params, xs_array[0].shape[1], fe.shape[1]]
        # 返回一个新的EffiDG实例，使用提供的配置和模型参数进行初始化。
        return EffiDG(cfg_params, model_params)

    # 2.3 先验分布
    # @property：定义一个属性方法prior_dist，用于返回先验分布。
    @property
    def prior_dist(self):
        # 计算先验分布的标准差对数，分别对应于u和z。
        [scale_u, scale_z] = [
            self.ln_sigma_u.T * torch.ones([1, self.dim_xs], device=device),
            self.ln_sigma_z.T * torch.ones([1, self.dim_xs], device=device)
        ]
        # 构造协方差矩阵的对角线部分。
        cov_ldiag = torch.cat([scale_u, scale_z], 1)
        # 初始化协方差因子和均值为零。
        [cov_factor, mu] = [
            torch.zeros_like(cov_ldiag)[:, :, None],
            torch.zeros_like(cov_ldiag)
        ]
        # 创建一个低秩多元正态分布实例，作为先验分布。
        prior_dist = LowRankMultivariateNormal(mu, cov_factor, torch.exp(2 * cov_ldiag))
        return prior_dist
    
    # 2.4 effidg核心前向传播[本质上利用贝叶斯后验（低质重构分布）拟合出来分布，固定特征*随机特征来重构包内特征表示并取均值后计算包预测标签]，接收输入数据xs=[350-包数量,5-类别数量,40-采样数量],计算出包预测z_bag=[350-包数量,5-类别数量,40-采样数量]
    def forward(self, xs):
        # 从后验分布中抽取beta_u和beta_z，分别对应于随机效应和固定效应。
        # beta_u[128-分布靠前的特征,5-类别数量,40-self.nr_samples个采样数] 是前self.nr_input个样本是随机特征，用于建模 实例级不确定性（如噪声、混杂因子），希望其稳定，代表“真实”信号，beta_u 的后验方差较大，采样结果波动明显 → 体现“随机性”。beta_u 被当作辅助噪声项或校正项，只在训练时引入随机性以提升鲁棒性，测试时虽用均值但重要性较低。
        # beta_z[128-分布靠后的特征,5-类别数量,40-self.nr_samples个采样数] 是最后self.nr_input个样本作为固定特征。用于建模 包级/全局判别特征（如类别原型、共享权重），beta_z 的后验方差可能被约束得很小（例如通过先验或正则化），导致其采样结果几乎不变 → 近似“固定。beta_z被当作模型的核心可学习参数（类似传统神经网络权重），用于最终分类；
        beta_u, beta_z = self.posterior.get_beta(self.nr_samples, self.predict_flag)

        # b=[1-beta_z维度均值,5-类别数,40-采样数]计算beta_z(固定效应)的平方根平均值，用于归一化，表示数据集的维度均值
        b = torch.sqrt((beta_z**2).mean(0, keepdim=True))
        # eta=[128-beta_z的特征维度,5-类别数,40-采样数]，归一化beta_z，得到eta【标准化的固定效应】。
        eta = beta_z / b 
        # Calculate importance weights of instances
        # x=[N-包中实例的总数(包的这维去掉，拼到包内实例这一维度), D-特征维度];i=[实例归属包的编号]；i_ptr=[包的起始位置]
        # 使用setup_scatter函数准备输入数据，返回特征矩阵x、索引i和指针i_ptr。
        x, i, i_ptr = setup_scatter(xs)
        # _w=[i=包中实例的总数,p-5-类别数量,s-40-self.nr_samples个采样数]=[去包实例数*特征维度]*[特征维度*类别*采样数(随机部分beta_u)]计算实例的重要性权重，使用爱因斯坦求和约定。
        _w = torch.einsum("iq,qps->ips", x, beta_u) # x.shape -> (i,q), beta_u.shape -> (q,p,s), _w.shape -> (i,p,s)
        # w=应用softmax函数，确保权重和为1。
        w = scatter_softmax(_w, i, dim=0)
        # t=[i=包中实例的总数,p-5-类别数量,s-40-self.nr_samples个采样数]=[去包实例数*特征维度]*[特征维度*类别*采样数(固定部分beta_z)] 计算特征与归一化后的beta_z的乘积。
        t = torch.einsum("iq,qps->ips", x, eta)
        # z_bag=[350-包数量,5-类别数量,40-采样数量]根据i_ptr对每个包中的元素进行合并（权重相乘再相加），得到综合的包标签。[相当于随机性的和固定性的在包内都考虑进去，并包内实例合并掉合并成了包级别的标签预测结果]
        z_bag = segment_add_csr(w * t, i_ptr) # summarizing elements based on (i_ptr) for each bag 

        # 计算汇总包分类结果的均值和标准差。
        mean, std = [z_bag.mean(0), z_bag.std(0)]
        # 如果标准差中有NaN值，则将其替换为1，避免除零错误。
        if std.isnan().any():
            std = 1
        # z_bag=[350-包数量,5-类别数量,40-采样数量] 包标签进行标准化。
        z_bag = b * (z_bag - mean) / std

        # 第二个只有固定特征来做graph
        # z_bag=[350-包数量,5-类别数量,40-采样数量]根据i_ptr对每个包中的元素进行合并（权重相乘再相加），得到综合的包标签。[相当于随机性的和固定性的在包内都考虑进去，并包内实例合并掉合并成了包级别的标签预测结果]
        fix_bag = segment_add_csr(t, i_ptr) # summarizing elements based on (i_ptr) for each bag 

        # 计算汇总包分类结果的均值和标准差。
        fix_mean, fix_std = [fix_bag.mean(0), fix_bag.std(0)]
        # 如果标准差中有NaN值，则将其替换为1，避免除零错误。
        if fix_std.isnan().any():
            fix_std = 1
        # z_bag=[350-包数量,5-类别数量,40-采样数量] 包标签进行标准化。
        fix_bag = b * (fix_bag - fix_mean) / fix_std

        # 返回标准化后的包标签结果。
        return z_bag,fix_bag
    
    # 2.5 损失函数
    # 定义损失函数计算方法，接收四个参数：u（随机效应）、fe（固定效应）、s（标签）和kld_w（KL散度权重）。
    def calculate_loss(self, u, fe, s, kld_w):
        # Equation (3)
        # logits[350-包数量,5-标签数,40-采样数]=计算综合预测值，结合fe固定效应=[350-包数量,1-标签]*self.alpha=[1,5]=[350,5,1]和u(包预测结果)随机效应=[350-包数量,5-类别数,40-采样数]。
        logits = fe.mm(self.alpha).unsqueeze(2) + u
        # logits_d[350-包数量,40-采样数,5-标签数]=调整逻辑值的维度顺序，便于后续计算。
        logits_d = logits.permute(0, 2, 1)
        # calculate components of loss function
        # link_func采样后类别的损失汇总平均值（分布拟合的越好，采样的值越接近）。计算似然函数的部分，使用logits_d和标签s进行加权求和。logits_d[350-包数量,40-采样数,5-标签数]*s.unsqueeze(1).expand(-1, self.nr_samples, -1)[350-包数量,40-采样数,5-标签数]
        link_func = torch.sum(
            torch.sum(
                logits_d * s.unsqueeze(1).expand(-1, self.nr_samples, -1), 
                dim=[1, 2]
                )
            ) / s.shape[0]
        # Equation (7)
        # 获取后验分布。
        posterior_dist = self.posterior.distribution
        # kld=计算KL散度，衡量后验分布与先验分布之间的差异。
        kld = kl_divergence(posterior_dist, self.prior_dist)
        # 计算KL散度项（各标签类别都要有1个），乘以权重并归一化。
        kld_term = kld_w * kld.sum() / s.shape[0]
        # Equation (6)
        # 组合似然函数和KL散度项，得到最终的损失值。
        loss = - link_func + kld_term

        # 构造一个字典，包含损失值loss、联接函数似然函数你和损失值和先验和后验损失的KL散度值。
        res_dict = {
            "loss": round(loss.item(), 4), 
            "ll": round(link_func.item(), 4), 
            "kld": round(kld_term.item(), 4)
        }
        return loss, res_dict
    
    # 2.6 目标函数（核心计算部分）
    # 定义目标函数计算方法，接收四个参数：data（输入特征数据）[350-包数量,48-包内实例数、784-特征数量]、fe（固定效应）、s_label（包伪标签）和ratio_kld（KL散度权重）。
    def calculate_obj(self, data, fe, s_label, ratio_kld):
        # 使用特征提取器处理输入数据，得到特征表示xs。xs和data都是带有包结构的实例，只不过在特征抽取的时候把包结构去掉了来抽取特征。
        xs = self.fea_extractor(data)
        # 核心计算-调用前向传播方法，计算包标签预测结果z_b。
        z_b,fix_bag = self(xs)
        # 计算损失值和生成结果字典。入参：z_b包预测，fe固定值，s_label偏标签，返回总体损失值和每个分项的损失值字典
        loss, res = self.calculate_loss(z_b, fe, s_label, kld_w=ratio_kld)
        # 返回loss总损失值、res包含损失值、似然函数值和KL散度值的字典，和z_b包预测标签
        return loss, res, z_b,fix_bag
    
    # 2.7 更新标签
    # 定义标签更新方法，接收三个参数：s（原始标签）、zb（预测结果）和w_conf（置信权重）。
    def regenerate_s(self, s, zb, w_conf):
        # 创建一个与S形状相同的全零张量，表示候选标签。
        s_candiate = torch.zeros(s.shape).to(device)
        # 将候选标签中大于0的位置设为1。
        s_candiate[s > 0] = 1
        # s_can_r=[350包数量,5类别数量，40采样数量] 扩展候选标签的维度，便于后续计算。
        s_can_r = s_candiate.unsqueeze(2).expand(-1, -1, self.nr_samples).detach()
        # s_pred=[350包数量,5类别数量] 计算候选的预测标签的值，并使用softmax函数进行归一化。
        s_pred = torch.mean(torch.softmax(zb.detach(), dim=1) * s_can_r, dim=2)
        # 将原始标签从计算图中分离出来，避免梯度传播。
        s = s.detach()
        # new_s，根据置信权重更新标签，结合原始标签和预测标签。w_conf会越来越小，新的预测作用会越来越小。
        new_s = w_conf * s + (1. - w_conf) * s_pred
        # 返回更新后的标签。
        return new_s
    
    # 2.7 获取预测标签
    # 定义标签更新方法，接收三个参数：s（原始标签）、zb（预测结果）和w_conf（置信权重）。
    def mean_predict(self, zb):
        # s_pred=[350包数量,5类别数量] 计算候选的预测标签的值，并使用softmax函数进行归一化。
        s_pred = torch.mean(torch.softmax(zb.detach(), dim=1), dim=2)
        return s_pred
    
    # 2.8 训练过程（训练入口）
    # 定义训练方法，接收三个参数：train_loader（数据加载器）、num_bags（袋子数量）和weight_list（权重列表）。
    def train(self, train_loader, num_bags, weight_list):


        # 从配置对象中获取学习率、训练轮数和正则化系数。
        [lr_value, n_epochs, reg_value] = [self.args.lr, self.args.epochs, self.args.reg]
        # optimizer = 创建一个SGD优化器，用于更新模型参数。
        optimizer = optim.SGD(self.parameters(), lr=lr_value, momentum=0.9,
                              nesterov=True, weight_decay=reg_value)
        X_recon = None # 包X的图重构特征x_recon
        Y_recon = None # 包Y的图重构标签y_recon
        for epoch in range(n_epochs):
            # 初始化参数
            aggre_bag_x_list = []#[None] * len(train_loader.dataset) # 包X特征聚合  # 
            aggre_fix_bag_list  = [] # 固定特征包X特征聚合
            z_bag_predict_label_list = []#[None] * len(train_loader.dataset) # 预测标签
            #  topk_bag_data_list = []
            partial_bag_lab_list = []# [None] * len(train_loader.dataset) # 包的偏标签
            #true_bag_lab_list = []
            # 遍历数据加载器，获取每个批次的数据。[这里的数据集可以并行,batch_size=350]
            for step, (x, s, _, cov_b, idx) in enumerate(train_loader):
                # 计算当前轮数。 
                i_epoch = epoch + 1
                # 计算KL散度权重，根据批次大小和包数量进行调整。
                ratio_kld = len(x) / num_bags
 
                # 计算目标函数核心计算逻辑、计算损失+新包偏标签z_bag。
                # 其中入参：实例特征x，固定效应权重cov_b,旧偏标签s，散度率ratio_kld
                loss, res, z_bag,fix_bag = self.calculate_obj(x, cov_b, s, ratio_kld)
                # 用流行学习计算获取相关数据
                # 更新包标签（根据一定比例新的标签+一定比例老标签，100%用新标签计算结果）。
                partial_bag_lab_list.extend([s[id] for id in idx])
                aggre_bag_x_list.extend([z_bag[id] for id in idx])  
                aggre_fix_bag_list.extend([fix_bag[id] for id in idx])  
                predict = self.mean_predict(z_bag)
                #predict = self.regenerate_s(s, z_bag, 1)
                z_bag_predict_label_list.extend([predict[id] for id in idx])  
                # 记录当前轮数和步数。
                res["epoch"], res["step"] = i_epoch, step
                # 如果当前是第一轮或每10轮打印一次日志。
                if epoch == 0 or (i_epoch) % 10 == 0:
                    # 记录损失信息。
                    logging.info("Loss Dict: {}".format(res))
                    print("Loss Dict: {}".format(res))

                # Dynamic Disambiguation
                # 获取当前轮数的权重（随着epoch会越来越小）。
                alpha_value = weight_list[epoch]
                # 增加流行学习权重
                if(Y_recon is None):
                    z_bag = z_bag
                else:
                    z_bag = (1-self.args.lambda_recon_rate)*z_bag + self.args.lambda_recon_rate*torch.tensor(Y_recon[idx]).to(device).unsqueeze(2).expand(-1, -1, self.nr_samples).detach()
                # 更新包标签（根据一定比例新的标签+一定比例老标签，新的权重受alpha_value影响）。
                new_s = self.regenerate_s(s, z_bag, alpha_value)
                # new_s=将更新后的偏标签写入数据集作为新的伪标签。
                train_loader.dataset.partial_bag_lab[idx] = new_s
                # if(Y_recon is None):
                #     train_loader.dataset.partial_bag_lab[idx] = new_s
                # else:
                #     train_loader.dataset.partial_bag_lab[idx] = (1-alpha_value*self.args.lambda_recon_rate)*new_s + alpha_value*self.args.lambda_recon_rate*torch.tensor(Y_recon[idx]).to(device)


                # 清空梯度。
                optimizer.zero_grad()
                # 反向传播，计算梯度。
                loss.backward()
                # 更新模型参数。 
                optimizer.step()
                # #partial_bag_lab_list.insert(batch_idx, np.copy(np.squeeze(new_label, axis=0)))  # 先用原本的 new_label已经更新了数据
                #true_bag_lab_np = true_bag_lab.cpu().numpy()
                #true_bag_lab_list.insert(batch_idx, np.copy(true_bag_lab_np))
            aggre_bag_x_list=torch.stack(aggre_bag_x_list,dim =0)
            aggre_fix_bag_list =torch.stack(aggre_fix_bag_list,dim=0)
            z_bag_predict_label_list=torch.stack(z_bag_predict_label_list,dim =0)
            partial_bag_lab_list=torch.stack(partial_bag_lab_list,dim =0)




            # 流行学习部分
            if epoch == 1 or (epoch) % 10 == 0:
                print('Epoch: {}, Train loss: {:.4f}.'.format(epoch, loss))
            if (epoch) % 30 == 1:
                print('Epoch: {}, deepm manifold update label started...'.format(epoch))
                # 数据格式变化为numpy
                aggre_bag_x_list = aggre_bag_x_list.reshape(aggre_bag_x_list.size(0),-1)#aggre_bag_x_list.permute([0, 2, 1]) # 移动顺序位置
                aggre_fix_bag_list = aggre_fix_bag_list.reshape(aggre_fix_bag_list.size(0),-1)
                aggre_bag_x_list_np = aggre_bag_x_list.cpu().detach().numpy() # 包X聚合
                aggre_fix_bag_list_np = aggre_fix_bag_list.cpu().detach().numpy() # 包X聚合
                partial_bag_lab_list_np = partial_bag_lab_list.cpu().detach().numpy() #偏标签
                z_bag_predict_label_list_np = z_bag_predict_label_list.cpu().detach().numpy() # 预测结果聚合
                aggre_bag_x_list_np = np.array(aggre_bag_x_list_np)
                aggre_fix_bag_list_np = np.array(aggre_fix_bag_list_np)
                partial_bag_lab_list_np = np.array(partial_bag_lab_list_np)
                z_bag_predict_label_list_np = np.array(z_bag_predict_label_list_np)
                #topk_bag_data_list=np.array(topk_bag_data_list)
                # manifold 模型计算
                # print("manifold_method1:lablemanifold....")
                # X_recon, Y_recon = lablemanifold(aggre_bag_data_tensor, partial_bag_lab_tensor.view(len(train_loader), -1), 10)
                print("deepm:build_label_manifold....")
                # 5. 初始化标签预测和标签传播的标签分布
                # Y_pred_np：初始化为 Y_P_train，表示当前的标签预测。
                # Y_recon：初始化为 Y_P_train，表示标签传播后的标签分布。
                # if Y_P_train is None:
                #     Y_P_train = partial_bag_lab_list
                if Y_recon is None:
                    Y_recon = partial_bag_lab_list_np
                # 6. 构建图结构和估计标签相关性矩阵
                # Wn：通过 build_graph 函数构建归一化的邻接矩阵，表示图中节点之间的相似度。
                #X_recon = aggre_bag_x_list_np
                #todo 两者可以替换交替
                X_recon =aggre_fix_bag_list_np
                X_bag_train = X_recon
                Y_P_train = partial_bag_lab_list_np #train_loader.dataset.train_true_bag_lab_list
                #Y_P_train, Y_recon, Wn, L, X_recon = deepm_train_global_prepare(args, X_bag_train, Y_P_train)
                Y_P_train, Y_recon, Wn, L, X_recon = deepm_train_global_prepare(args, X_bag_train, Y_P_train)

                    
        # 设置预测标志为True，表示训练完成。
        self.predict_flag = True
        return 
    
    # 2.9 预测方法
    # 装饰器，确保推理模式下不计算梯度，提高效率。
    @torch.inference_mode()
    def predict(self, xs):
        self.nr_samples = None
        # Equation (9)
        # 定义预测方法，接收输入数据xs。 
        xs = self.fea_extractor(xs)
        # 调用前向传播方法，计算预测结果，并压缩维度。
        pred,_ = self(xs)
        s_hat = pred.squeeze(2)
        # 对预测结果应用softmax函数，得到概率分布。
        s_logits = torch.softmax(s_hat, dim=1)
        # 选择概率最大的类别作为预测结果。
        y_pred = torch.max(s_logits.data, 1)[1]
        # 返回预测结果。
        return y_pred

