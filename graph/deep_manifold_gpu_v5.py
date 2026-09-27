# 用于数值计算。
import numpy as np
# 提供数学函数。
import math
# warnings：用于控制警告信息的显示。
import warnings
# argparse：用于解析命令行参数。
import argparse
# torch 和 torch.nn.functional：PyTorch 深度学习框架，用于构建和训练神经网络。
import torch
import torch.nn.functional as F
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


#from ELIMIPL.main_manifold import X_recon
from utils import *
from deep_manifold_models import *
# faiss：Facebook AI 提供的高效相似性搜索库，特别适用于大规模数据集的 kNN 搜索。
import faiss
from faiss import normalize_L2
# scipy 和 scipy.stats：科学计算库，用于统计分析。
import scipy
import scipy.stats
import time
# sklearn.preprocessing 和 sklearn.neighbors：用于数据预处理和 kNN 图构建。
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph
# tqdm：用于显示进度条。
from tqdm import tqdm
# os 和 json：用于文件操作和 JSON 解析。
import os
import json
# 2. 忽略警告信息
warnings.filterwarnings("ignore")

# 3. 估计标签相关矩阵[计算不同标签再样本中共线出现在总它的总共线中的比例，计算该标签相关性。这个矩阵可以用来衡量不同标签之间的关联程度]
# 功能：该函数用于估计标签之间的相关性矩阵L。输入 Y_P 是一个形状为 (n, num_class) 的矩阵，表示每个样本的标签分布。
# 步骤：
# （1）.初始化一个全零矩阵R，用于存储标签之间的相关性。
# （2）.对于每一对标签i和j，如果它们不是同一个标签（即 i!=j），计算它们之间的相关性。【相关性是通过两个标签向量的点积除以它们的总和】来计算的。
# （3）.如果某个标签的总和为零，为了避免除零错误，将其相关性设为 1e-5。
# （4）.最后，通过归一化操作将 R 转换为对称正定矩阵 L，并返回 L。
def estimating_label_correlation_matrix(Y_P):
	num_class = Y_P.shape[1]  # 样本类别
	n = Y_P.shape[0]  # 样本数

	R = np.zeros((num_class, num_class)) #R矩阵 L*L，其中L是标签的数量。这个矩阵将用于存储标签之间的相关性值。
	for i in range(num_class):
		for j in range(num_class):
			# 对角线元素：当 i == j 时，R[i][j] 被设置为 0。这表示同一个标签之间的相关性为 0（即不考虑标签自身的相关性）
			if i == j:
				R[i][j] = 0
			# 非对角线元素：当i!=j时，计算标签i和标签j之间的相关性。具体计算公式为：
			else:
				if np.sum(Y_P[:, i]) == 0 and np.sum(Y_P[:, j]) == 0 :
					R[i][j] = 1e-5 # 设置一个很小的数，防止除不尽。avoid divide zero error
				else:
					# Y_P[:, i] 表示所有样本中标签i的标注情况（0 或 1），Y_P[:, j] 表示所有样本中标签j 的标注情况（0 或 1）。
					# Y_P[:, i].dot(Y_P[:, j]) 计算的是标签i和标签j同时出现的样本数。
					# Y_P[:, i].sum() 和 Y_P[:, j].sum() 分别是标签i和标签j 出现的总次数。通过除以两个标签出现次数之和，归一化了它们的共现频率。
					R[i][j] = Y_P[:, i].dot(Y_P[:, j]) / (Y_P[:, i].sum() + Y_P[:, j].sum()) # 对应到论文公式中求a^y_ij并求A_y
	D_1_2 = np.diag(1. / np.sqrt(np.sum(R, axis=1)))  # 计算的是每一行的和，表示每个标签与其他所有标签的相关性之和，将这些值放在对角线上，构建一个对角矩阵。
	L = D_1_2.dot(R).dot(D_1_2) # D_1_2对R点乘，然后再对D_1_2点乘。
	# np.nan_to_num 函数函数用于将数组中的 NaN 和无穷大值替换为合理的数值。
	# 具体来说： NaN 被替换为 0。正无穷大被替换为最大有限浮点数。负无穷大被替换为最小有限浮点数。
	L = np.nan_to_num(L)
	# 返回归一化的标签相关性矩阵L，它是一个L×L 的矩阵，表示标签之间的相关性。
	return L

# 4. 构建图（版本2）【K近邻构建领接矩阵构图，向量距离越近度越少权重越高，从而计算X相似】
# 功能：该函数用于构建样本之间的相似性图，使用 kNN 算法。它是一个备用版本，通常不使用。
# 步骤：
# 使用 kneighbors_graph 函数构建 kNN 图，mode='distance' 表示边权重为样本之间的距离（权重是距离）。
# 将距离转换为相似度（通过指数衰减函数），并对称化图。将图的邻接矩阵W转换为一个对称的、归一化的邻接矩阵Wn
# 归一化图，确保每个节点的度数为 1。

def build_graph_v2(X, k=10):
	if not args.no_verbose:
		print('Building Graph - V2 is not used')
	# (a)KNN图检索,其中把特征传进来，k是近邻数量 kNN search for the graph
	W = kneighbors_graph(X, k, mode='distance', include_self=False, metric='euclidean')

	# (b)应用高斯核函数（Gaussian Kernel Function），其数学表达式为：exp(-距离^2/(2*标准差的平方))
	# 其中 W 是一个稀疏矩阵，通常使用 scipy.sparse 库中的稀疏矩阵格式（如 csr_matrix）来表示图的邻接矩阵，
	# 其中 W.data 是一个一维数组，包含了稀疏矩阵中所有非零元素的值。这些值在初始情况下通常是 k 近邻搜索得到的距离值。
	# 其中 W.data ** 2：这一步将 W.data 中的所有距离值平方。平方操作是为了放大距离的差异，使得距离较近的点之间的相似度更高，而距离较远的点之间的相似度更低
	W.data = np.exp(- W.data ** 2 / (2 * 1))

	# (c)对称化：确保图的邻接矩阵是对称的，适用于无向图。这一步将W和W.T相加，确保邻接矩阵是对称的。因为在无向图中，如果节点i和节点j之间有边，则𝑊_ij=W_ji。
	# 通过加上转置矩阵，可以保证即使原始的W不是对称的，最终的W也是对称的。
	W = W + W.T
	# (d)去除自环：确保邻接矩阵中没有自环（即节点与自身之间的边）。在图传播中，通常不希望节点与自身相连，因此需要移除对角线上的非零值。
	# 将 W 减去这个对角矩阵，实际上相当于将 W 的对角线元素全部设为 0。
	W = W - scipy.sparse.diags(W.diagonal())
	# (e)用度构建对角矩阵D
	# 数组S实际上是每个节点的度（Degree），即每个节点连接的邻居数量。W.sum(axis=1) 沿着每一行求和，返回一个包含每行元素之和的一维数组。
	S = W.sum(axis=1)
	# 去除0，如果某个节点的度为 0（即该节点没有邻居），则将其度设置为 1:这一步是为了防止后续计算中出现除以零的情况。
	S[S == 0] = 1
	# 计算度矩阵的平方根倒数：用于后续的归一化操作。
	D = np.array(1. / np.sqrt(S))
	# 构建度矩阵的平方根倒数矩阵：D 是一个对角矩阵，用于后续的归一化操作。其中 D.reshape(-1)将矩阵转成数组
	D = scipy.sparse.diags(D.reshape(-1))
	# (f)这一步是将邻接矩阵 W 与度矩阵 D 进行两次相乘，得到归一化的邻接矩阵Wn
	# 这种归一化方式称为对称归一化，它使得每个节点与其邻居之间的相似度权重被节点的度所调整，从而更好地反映节点之间的相对重要性。
	Wn = D * W * D

	return Wn

# 5. 构建图（版本1，使用 Faiss）【K近邻构建领接矩阵构图，向量距离越近度越少权重越高，从而计算X相似】
# 功能：该函数使用 Faiss 库进行高效的 kNN 搜索，构建样本之间的相似性图。
# 用于构建一个基于 k-近邻（kNN）搜索的图，并对图的邻接矩阵进行归一化处理。这个图可以用于图神经网络（GNN）、谱聚类等任务中。
# 参数 ：
# X：形状为 (N, d) 的特征矩阵，其中 N 是样本数量，d 是每个样本的特征维度。
# k：每个节点的 k 近邻数，默认值为 10。
# args：包含一些配置参数的对象，例如是否打印日志信息（no_verbose）。
# 步骤：y
# 将输入数据 X 转换为 float32 类型，并构建 Faiss 的 GPU 索引。
# 对数据进行 L2 归一化，并将数据添加到索引中。
# 使用 Faiss 进行 kNN 搜索，获取每个样本的最近邻及其距离。
# 根据搜索结果构建稀疏相似性图 W，并对图进行归一化处理。
def build_graph(X, k=10, args=None):
	if not args.no_verbose:
		print('Building Graph - V1')
	# kNN search for the graph
	# 3. 数据预处理
	# 将特征矩阵X转换为float32类型，以节省内存并提高计算效率。
	X = X.astype('float32')
	# d是特征的维度，即X的第二维大小。
	d = X.shape[1]
	# 4. 初始化 FAISS GPU 索引
	# FAISS 是一个高效的相似性搜索库，特别适用于大规模数据集的近似最近邻搜索。
	# faiss.StandardGpuResources() 创建一个 GPU 资源对象，用于管理 GPU 内存和计算资源。
	# faiss.GpuIndexFlatConfig() 配置 GPU 索引，指定使用第 0 块 GPU。
	# faiss.GpuIndexFlatIP(res, d, flat_config) 创建一个基于内积（Inner Product, IP）的索引，用于快速查找最相似的 k 个邻居。
	res = faiss.StandardGpuResources()
	flat_config = faiss.GpuIndexFlatConfig()
	flat_config.device = 0
	index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index

	# 5. 归一化特征向量
	# normalize_L2(X)是一个函数调用，用于将特征向量X归一化为L2单位范数。这一步是为了确保内积计算的有效性，因为内积在归一化后的向量上表现更好。
	
	# 检查数据是否异常
	print(f"Before normalization - X stats: min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}, std={X.std():.6f}")
	print(f"X contains NaN: {np.isnan(X).any()}")
	print(f"X contains Inf: {np.isinf(X).any()}")
	print(f"X contains zeros: {(X == 0).all(axis=1).any()}")
	
	# 确保 X 是 numpy array, float32, 2D, C-contiguous
	if not X.flags['C_CONTIGUOUS']:
		X = np.ascontiguousarray(X, dtype=np.float32)
	else:
		X = X.astype(np.float32)  # 确保是 float32
	normalize_L2(X)
	
	# 检查归一化后的数据
	print(f"After normalization - X stats: min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}, std={X.std():.6f}")
	print(f"X contains NaN after norm: {np.isnan(X).any()}")
	print(f"X contains Inf after norm: {np.isinf(X).any()}")
	print(f"X L2 norms: min={np.linalg.norm(X, axis=1).min():.6f}, max={np.linalg.norm(X, axis=1).max():.6f}")
	# 6. 添加特征到索引
	# index.add(X) 将特征矩阵 X 添加到 FAISS 索引中，以便后续进行 kNN 搜索。
	# N 是样本的数量。
	# Nidx 是索引中存储的总样本数，理论上应该等于 N。
	index.add(X)
	N = X.shape[0]
	Nidx = index.ntotal

	# 7. 执行 kNN 搜索
	c = time.time()
	# index.search(X, k + 1) 对每个样本执行 kNN 搜索，返回每个样本的 k+1 个最近邻的距离 D 和对应的索引 I。这里 k+1 是为了包括样本自身（即每个样本的第一个最近邻是它自己）。
	D, I = index.search(X, k + 1) # D k近邻的权重值，I k近邻的节点索引值
	# time.time() 用于记录 kNN 搜索的时间，并在 args.no_verbose 为 False 时打印搜索时间。
	elapsed = time.time() - c
	if not args.no_verbose:
		print('kNN Search Time: ', elapsed)
	
	# 调试信息
	print(f"X shape: {X.shape}, k: {k}, N: {N}")
	print(f"D shape: {D.shape}, I shape: {I.shape}")
	print(f"D range: [{D.min():.4f}, {D.max():.4f}]")
	print(f"I range: [{I.min()}, {I.max()}]")
	print(f"First few I values: {I[:3, :]}")

	# 8.基于K近邻构建邻接矩阵 Create the graph
	# 去除自环：D[:, 1:] 和 I[:, 1:] 去掉了每个样本自身的距离和索引，只保留了真正的 k 个最近邻。
	# 距离立方：D[:, 1:] ** 3 将距离值的立方作为边权重。这一步的作用是放大距离差异，使得距离较近的节点之间的权重更高，而距离较远的节点之间的权重更低。
	D = D[:, 1:] ** 3
	I = I[:, 1:]
	
	# 过滤掉无效的-1索引 ——tk
	# print("test_I:",I)
	# valid_mask = (I != -1)
	# D = D * valid_mask  
	# I = I * valid_mask  
 
 
	# 构建稀疏矩阵：scipy.sparse.csr_matrix 用于创建一个稀疏的邻接矩阵 W。
	# D.flatten('F') 是权重值，row_idx_rep.flatten('F') 是行索引，I.flatten('F') 是列索引。shape=(N, N) 指定了矩阵的形状。
	# 行索引，作用：创建一个从 0 到 N-1 的整数数组，其中 N 是样本的数量。这个数组表示每个样本的索引。
	row_idx = np.arange(N)
	# row_idx_rep 为(N, k) 里面是每个样本的索引，非权重值
	# 列索引。np.tile 函数：np.tile 用于将一个数组沿指定的维度重复多次。
	# 具体来说，np.tile(array, (reps)) 会将 array 沿第一个维度重复 reps[0] 次，沿第二个维度重复 reps[1] 次，依此类推。
	# row_idx_rep 的每一行表示某个样本的索引，重复了 k 次。这是因为每个样本有 k 个邻居，所以我们需要为每个邻居分配一个对应的行索引。
	row_idx_rep = np.tile(row_idx, (k, 1)).T
	# D.flatten('F') D 是一个形状为 (N, k) 的矩阵，表示每个样本与其 k 个最近邻的距离，然后按列优先被展开
	# row_idx_rep 矩阵：如前所述，row_idx_rep 是一个形状为 (N, k) 的矩阵，每一行表示某个样本的索引，重复了 k 次。
	# flatten('F')：形状为 (N, k) 的矩阵 row_idx_rep，row_idx_rep.flatten('F') 会生成一个长度为 N * k 的一维数组，表示每条边的起始节点（即行索引）
	# I 矩阵：I 是一个形状为 (N, k) 的矩阵，表示每个样本的 k 个最近邻的索引。经过 I = I[:, 1:] 处理后，I 只包含真正的 k 个邻居的索引,并且按列被展平。
	# scipy.sparse.csr_matrix：这是一个用于创建压缩稀疏行（Compressed Sparse Row, CSR）格式的稀疏矩阵的函数。CSR 格式是一种高效的稀疏矩阵存储方式，特别适合快速的行访问和矩阵向量乘法。
	# W 是一个形状为 (N, N) 的稀疏矩阵。参数解释：
	# 第一个参数是一个元组 (data, (row, col))，其中：
	# data 是边的权重，即 D.flatten('F')，表示每条边的权重。
	# row 是边的起始节点索引，即 row_idx_rep.flatten('F')，表示每条边的行索引。
	# col 是边的目标节点索引，即 I.flatten('F')，表示每条边的列索引。
	# shape=(N, N)：指定稀疏矩阵的形状为 (N, N)，即图中有 N 个节点，邻接矩阵是一个 N x N 的矩阵。
	# 输出：W 是一个形状为 (N, N) 的稀疏矩阵，表示图中节点之间的连接关系。矩阵中的非零元素表示两个节点之间存在边，边的值为 D 中对应的距离权重。
	W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
	# 对称化：W = W + W.T 确保邻接矩阵是对称的，适用于无向图。
	W = W + W.T



	# 9. 归一化邻接矩阵 Normalize the graph
	# 去除自环：W = W - scipy.sparse.diags(W.diagonal()) 移除了邻接矩阵的对角线元素，确保没有自环。
	W = W - scipy.sparse.diags(W.diagonal())

	# todo 补充去除非0
	# W.data[W.data < 0] = 0  # 去除负值
	# W.eliminate_zeros()

	# 计算节点度：S = W.sum(axis=1) 计算每个节点的度（即每个节点连接的邻居数量）。
	S = W.sum(axis=1)
	# 避免除零错误：S[S == 0] = 1 防止后续计算中出现除以零的情况。
	S[S == 0] = 1
	# 构建度矩阵的平方根倒数：D = np.array(1. / np.sqrt(S)) 计算每个节点的度的平方根的倒数。
	D = np.array(1. / np.sqrt(S))
	D = scipy.sparse.diags(D.reshape(-1))
	# 归一化邻接矩阵：Wn = D * W * D 对邻接矩阵进行对称归一化，得到归一化的邻接矩阵𝑊n
	Wn = D * W * D

	return Wn
# 6new. 标签传播【标签的更新+特征更新】
# 功能：该函数实现了标签传播算法，通过图传播更新标签分布。
# 标签传播算法（Label Propagation Algorithm, LPA），用于通过图结构传播标签信息，并更新标签分布。该算法结合了图的邻接矩阵、初始标签分布、模型预测以及标签相关性矩阵，通过迭代优化标签分布。
# 参数说明：
# 输入：
# args：包含一些配置参数的对象，例如学习率 gamma。
# Wn：归一化的邻接矩阵，表示图中节点之间的相似度或连接强度。
# L：标签相关性矩阵，表示不同标签之间的关联性。
# Y_pred：模型预测的标签分布。
# Y_P_train：训练集中的初始标签分布（部分标记数据）。
# Z_current：当前的标签分布，通常初始化为 Y_P_train。
# alpha：控制基于图结构传播项的权重。
# eta：控制基于初始标签分布约束项的权重。
# beta：控制基于标签相关性约束项的权重。
# maxiter：最大迭代次数。
# 步骤：
# 将输入的标签分布 Z_current 和其他相关矩阵转换为 PyTorch 张量，并移动到 GPU 上。
# 在 maxiter 次迭代中，计算梯度并更新标签分布 Z_g。梯度由四个部分组成：
# alpha * (Z_g - W_matmul_Z_g)：基于图结构的传播项。
# eta * (Z_g - Y_P_train_g)：基于初始标签分布的约束项。
# 1 * (Z_g - Y_pred_g)：基于模型预测的约束项。
# beta * (Z_g - Z_g @ L_g)：基于标签相关性的约束项。
# 更新后的标签分布 Z 经过 Min-Max 归一化后返回。
def label_propagation_with_x_reconstruction(args, Wn, L, Y_pred, Y_P_train, Z_current, alpha, eta, beta, maxiter,X_train =None):

	# 2. 初始化标签分布
	# gamma 是学习率，控制每次更新标签分布时的步长。
	gamma = args.gamma  # learning rate
	# Z 初始化为 Y_P_train，即训练集中的初始标签分布。
	Z = Y_P_train

	# 3. 将输入数据转换为 PyTorch 张量并移动到 GPU
	# 将 Z、Y_P_train、Y_pred 和 L 转换为 PyTorch 的张量，并将它们移动到 GPU 上进行加速计算。
	# torch.from_numpy(...) 将 NumPy 数组转换为 PyTorch 张量。 .float() 将张量的数据类型转换为 float32，以节省内存并提高计算效率。.detach() 断开张量与计算图的连接，避免梯度追踪。.cuda() 将张量移动到 GPU 上。
	Z_g = torch.from_numpy(Z).float().detach().cuda()
	Y_P_train_g = torch.from_numpy(Y_P_train).float().detach().cuda()
	Y_pred_g = torch.from_numpy(Y_pred).float().detach().cuda()
	L_g = torch.from_numpy(L).float().detach().cuda()

	# 4. 标签传播迭代（todo  需要重构的代码）
	with torch.no_grad():
		for i in range(maxiter):
			# W_matmul_Z_g：计算图传播项 W⋅Z，即将当前标签分布 Z_g 通过图的邻接矩阵 Wn 进行传播。这里使用 Wn.dot(Z_g.cpu().numpy()) 将 Z_g 移回 CPU 进行矩阵乘法，然后再移回 GPU。
			# grad：计算梯度，梯度由四个部分组成：
			# 基于图结构的传播项：alpha * (Z_g - W_matmul_Z_g)，鼓励标签在图上平滑传播，使得相邻节点的标签分布更加相似。
			# 基于初始标签分布的约束项：eta * (Z_g - Y_P_train_g)，确保标签分布不会偏离初始标签分布太远。
			# 基于模型预测的约束项：1 * (Z_g - Y_pred_g)，确保标签分布接近模型的预测结果。
			# 基于标签相关性的约束项：beta * (Z_g - Z_g @ L_g)，利用标签相关性矩阵 L 来调整标签分布，使得相关标签之间的分布更加一致。
			# 更新标签分布：Z_g = Z_g - gamma * grad，根据计算的梯度更新标签分布 Z_g，步长由学习率 gamma 控制。
			W_matmul_Z_g = torch.from_numpy(Wn.dot(Z_g.cpu().numpy())).detach().cuda()
			# 论文15的公式（3）核心部分
			grad = alpha * (Z_g - W_matmul_Z_g) + eta * (Z_g - Y_P_train_g) + 1 * (Z_g - Y_pred_g) + beta * (Z_g - Z_g @ L_g)
			#tk  去掉L
			# grad = alpha * (Z_g - W_matmul_Z_g) + eta * (Z_g - Y_P_train_g) + 1 * (Z_g - Y_pred_g)  
			# 论文15的公式（3）核心部分 去掉标签流行学习的部分。
			#grad = alpha * (Z_g - W_matmul_Z_g) + eta * (Z_g - Y_P_train_g) + 1 * (Z_g - Y_pred_g) #+ beta * (Z_g - Z_g @ L_g)
			Z_g = Z_g - gamma * grad
		if X_train is not None:
			X_recon_new = torch.from_numpy(Wn.dot(torch.from_numpy(X_train))).detach().cuda() 
	# 归一化 使用 softmax 实现（推荐）
	Z_g = torch.softmax(Z_g, dim=1)
	# 5. 将更新后的标签分布移回 CPU 并转换为 NumPy 数组
	Z = Z_g.detach().cpu().numpy()
	# 6. Min-Max 归一化 todo 这个归一有问题，因为要消歧，综合为0
	# # 使用 MinMaxScaler 对标签分布 Z 进行 Min-Max 归一化，将每个标签的值缩放到 [0, 1] 区间内。这一步有助于确保标签分布的数值范围合理，避免极端值对后续任务的影响。
	# min_max_scaler = MinMaxScaler()
	# Z = min_max_scaler.fit_transform(Z)

	# 释放 GPU 内存缓存，防止内存泄漏，尤其是在多次调用该函数时，确保 GPU 内存得到有效管理。
	torch.cuda.empty_cache()

	# 返回更新后的标签分布
	return Z,X_recon_new

# 6. 标签传播【标签的更新】
# 功能：该函数实现了标签传播算法，通过图传播更新标签分布。
# 标签传播算法（Label Propagation Algorithm, LPA），用于通过图结构传播标签信息，并更新标签分布。该算法结合了图的邻接矩阵、初始标签分布、模型预测以及标签相关性矩阵，通过迭代优化标签分布。
# 参数说明：
# 输入：
# args：包含一些配置参数的对象，例如学习率 gamma。
# Wn：归一化的邻接矩阵，表示图中节点之间的相似度或连接强度。
# L：标签相关性矩阵，表示不同标签之间的关联性。
# Y_pred：模型预测的标签分布。
# Y_P_train：训练集中的初始标签分布（部分标记数据）。
# Z_current：当前的标签分布，通常初始化为 Y_P_train。
# alpha：控制基于图结构传播项的权重。
# eta：控制基于初始标签分布约束项的权重。
# beta：控制基于标签相关性约束项的权重。
# maxiter：最大迭代次数。
# 步骤：
# 将输入的标签分布 Z_current 和其他相关矩阵转换为 PyTorch 张量，并移动到 GPU 上。
# 在 maxiter 次迭代中，计算梯度并更新标签分布 Z_g。梯度由四个部分组成：
# alpha * (Z_g - W_matmul_Z_g)：基于图结构的传播项。
# eta * (Z_g - Y_P_train_g)：基于初始标签分布的约束项。
# 1 * (Z_g - Y_pred_g)：基于模型预测的约束项。
# beta * (Z_g - Z_g @ L_g)：基于标签相关性的约束项。
# 更新后的标签分布 Z 经过 Min-Max 归一化后返回。
def label_propagation(args, Wn, L, Y_pred, Y_P_train, Z_current, alpha, eta, beta, maxiter):

	# 2. 初始化标签分布
	# gamma 是学习率，控制每次更新标签分布时的步长。
	gamma = args.gamma  # learning rate
	# Z 初始化为 Y_P_train，即训练集中的初始标签分布。
	Z = Y_P_train

	# 3. 将输入数据转换为 PyTorch 张量并移动到 GPU
	# 将 Z、Y_P_train、Y_pred 和 L 转换为 PyTorch 的张量，并将它们移动到 GPU 上进行加速计算。
	# torch.from_numpy(...) 将 NumPy 数组转换为 PyTorch 张量。 .float() 将张量的数据类型转换为 float32，以节省内存并提高计算效率。.detach() 断开张量与计算图的连接，避免梯度追踪。.cuda() 将张量移动到 GPU 上。
	Z_g = torch.from_numpy(Z).float().detach().cuda()
	Y_P_train_g = torch.from_numpy(Y_P_train).float().detach().cuda()
	Y_pred_g = torch.from_numpy(Y_pred).float().detach().cuda()
	L_g = torch.from_numpy(L).float().detach().cuda()

	# 4. 标签传播迭代（todo  需要重构的代码）
	with torch.no_grad():
		for i in range(maxiter):
			# W_matmul_Z_g：计算图传播项 W⋅Z，即将当前标签分布 Z_g 通过图的邻接矩阵 Wn 进行传播。这里使用 Wn.dot(Z_g.cpu().numpy()) 将 Z_g 移回 CPU 进行矩阵乘法，然后再移回 GPU。
			# grad：计算梯度，梯度由四个部分组成：
			# 基于图结构的传播项：alpha * (Z_g - W_matmul_Z_g)，鼓励标签在图上平滑传播，使得相邻节点的标签分布更加相似。
			# 基于初始标签分布的约束项：eta * (Z_g - Y_P_train_g)，确保标签分布不会偏离初始标签分布太远。
			# 基于模型预测的约束项：1 * (Z_g - Y_pred_g)，确保标签分布接近模型的预测结果。
			# 基于标签相关性的约束项：beta * (Z_g - Z_g @ L_g)，利用标签相关性矩阵 L 来调整标签分布，使得相关标签之间的分布更加一致。
			# 更新标签分布：Z_g = Z_g - gamma * grad，根据计算的梯度更新标签分布 Z_g，步长由学习率 gamma 控制。
			W_matmul_Z_g = torch.from_numpy(Wn.dot(Z_g.cpu().numpy())).detach().cuda()
			# 论文15的公式（3）核心部分
			grad = alpha * (Z_g - W_matmul_Z_g) + eta * (Z_g - Y_P_train_g) + 1 * (Z_g - Y_pred_g) + beta * (Z_g - Z_g @ L_g)
			# 论文15的公式（3）核心部分 去掉标签流行学习的部分。
			#grad = alpha * (Z_g - W_matmul_Z_g) + eta * (Z_g - Y_P_train_g) + 1 * (Z_g - Y_pred_g) #+ beta * (Z_g - Z_g @ L_g)
			Z_g = Z_g - gamma * grad

	# 归一化 使用 softmax 实现（推荐）
	Z_g = torch.softmax(Z_g, dim=1)
	# 5. 将更新后的标签分布移回 CPU 并转换为 NumPy 数组
	Z = Z_g.detach().cpu().numpy()
	# 6. Min-Max 归一化 todo 这个归一有问题，因为要消歧，综合为0
	# # 使用 MinMaxScaler 对标签分布 Z 进行 Min-Max 归一化，将每个标签的值缩放到 [0, 1] 区间内。这一步有助于确保标签分布的数值范围合理，避免极端值对后续任务的影响。
	# min_max_scaler = MinMaxScaler()
	# Z = min_max_scaler.fit_transform(Z)

	# 释放 GPU 内存缓存，防止内存泄漏，尤其是在多次调用该函数时，确保 GPU 内存得到有效管理。
	torch.cuda.empty_cache()

	# 返回更新后的标签分布
	return Z

# 7. 模型训练与评估
# 输入：
# args：包含配置参数的对象，例如学习率、批量大小、是否使用标签传播等。
# data：一个包含训练数据和测试数据的元组 (X_train, Y_train, Y_P_train, X_test, Y_test)。
# model：要训练的神经网络模型。
# optimizer：优化器，用于更新模型参数。
# training_now：布尔值，表示是否正在进行训练（默认为 True）。
# eval_every：每隔多少个 epoch 进行一次评估，默认为 5。
def run_model(args, data, model, optimizer, training_now=True, eval_every=5):
	res = None  # 定义返回值 define the return value
	# res_log 和 overall_loss_log：用于记录训练过程中的日志信息，分别记录详细的损失和每个epoch的总体损失。
	res_log = np.zeros(args.epochs * 150)
	overall_loss_log = np.zeros(args.epochs)

	# 3.解包数据
	# X_train：训练集的特征矩阵。
	# Y_train：训练集的真实标签（可能未使用）。
	# Y_P_train：训练集的初始标签分布（部分标记数据）。
	# X_test：测试集的特征矩阵。
	# Y_test：测试集的真实标签。
	# batch_size：每次训练时使用的批量大小。
	# num_class：标签的数量。
	X_train, Y_train, Y_P_train, X_test, Y_test = data
	# 4. 计算每个 epoch 的迭代次数
	batch_size = args.batch_size
	num_class = Y_P_train.shape[1]
	# def data and frequently used args

	# number of iteration times per epoch
	# iter_per_epoch：每个 epoch 中的迭代次数，确保所有训练样本都被遍历到。使用 math.ceil 以处理最后一个不完整的批次。
	iter_per_epoch = int(math.ceil(X_train.shape[0] / batch_size))
	# iter_per_epoch = 1 # without batch

	loss_pred = 0

	# counter
	iter_cnt = 0

    # 5. 初始化标签预测和标签传播的标签分布
	# Y_pred_np：初始化为 Y_P_train，表示当前的标签预测。
	# Y_lp_np：初始化为 Y_P_train，表示标签传播后的标签分布。
	Y_pred_np = Y_P_train
	Y_lp_np = Y_P_train

	# 6. 构建图结构和估计标签相关性矩阵
	# Wn：通过 build_graph 函数构建归一化的邻接矩阵，表示图中节点之间的相似度。
	Wn = build_graph(X_train, k=args.neighbors_num, args=args)
	# L：通过 estimating_label_correlation_matrix 函数估计标签相关性矩阵，表示不同标签之间的关联性。
	L = estimating_label_correlation_matrix(Y_P_train)

	best_score = 0

	# 7. 训练循环
	for e in range(args.epochs):
		# 打乱训练数据的索引 shuffle indices
		train_indicies = np.arange(X_train.shape[0])
		np.random.shuffle(train_indicies)

		# 标签传播：如果 args.using_lp 为 True，则调用 label_propagation 函数执行标签传播算法，更新标签分布 Y_lp_np。
		if args.using_lp:
			maxiter = args.maxiter
			Y_lp_np = label_propagation(args, Wn, L, Y_pred_np, Y_P_train, Y_lp_np
				, alpha=args.alpha, eta=args.eta, beta=args.beta, maxiter=maxiter) #标签更新了，也就是Z更新了
		
		# 8. 批量训练
		# keep track of losses
		# 设置模型为训练模式：model.train()。
		model.train()

		for i in range(iter_per_epoch):
			# 清除梯度：optimizer.zero_grad()，防止梯度累积。
			optimizer.zero_grad()

			# 计算索引位置和批次内的索引序号
			start_idx = (i * batch_size) % X_train.shape[0]
			idx = train_indicies[start_idx: start_idx + batch_size]
			# idx = train_indices # without batch

			# 获取当前批次的数据：根据索引 idx 获取当前批次的特征 X 和标签 Y_P 或 Y_lp。
			# get minibatch indices
			# randomly select a mini-batch of data
			X = torch.from_numpy(X_train[idx, :]).float().detach().cuda()
			Y_P = torch.from_numpy(Y_P_train[idx, :]).float().detach().cuda()
			Y_lp = torch.from_numpy(Y_lp_np[idx, :]).float().detach().cuda()
			X.requires_grad = False
			Y_P.requires_grad = False
			Y_lp.requires_grad = False

			# 前向传播：将特征 X 输入模型，获取预测结果 Y_pred。
			Y_pred = model.forward(X)

			# 计算损失：根据是否使用标签传播，选择不同的损失函数：
			# 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
			# 否则，使用二元交叉熵损失 F.binary_cross_entropy(Y_pred, Y_P)。
			if args.using_lp:
				loss = F.mse_loss(Y_pred, Y_lp)
			else:
				loss = F.binary_cross_entropy(Y_pred, Y_P)

			# 反向传播：loss.backward()计算梯度。
			loss.backward()
			# 更新参数：optimizer.step() 根据计算的梯度更新模型参数。
			optimizer.step()
			# approximating candidate labels

		# 9. 更新test的标签预测
		# 在每个epoch结束后，使用test函数对训练集进行预测，更新 Y_pred_np。
		Y_pred_np = test(args, X_train, num_class, model)

		# 10. 评估模型
		Z = Y_lp_np
		# 设置模型为评估模式：model.eval()。
		model.eval()

		# testing
		if args.no_verbose:
			eval_every = 1

		# 测试模型：每隔 eval_every 个 epoch，在测试集上进行预测，并调用 evaluate 函数评估模型性能，
		# 计算 r_loss（排名损失）、h_loss（汉明损失）和 ap（平均精度）。
		if e % eval_every == 0:
			Y_pred = test(args, X_test, num_class, model)
			r_loss, h_loss, ap = evaluate(Y_test, Y_pred, threshold=args.threshold)

			# 打印日志：如果 args.no_verbose 为 False，则打印当前 epoch 的评估结果。
			if not args.no_verbose:
				print('Epoch %d: r_loss %.4f, h_loss %.4f, ap %.4f' 
					% (e, r_loss, h_loss, ap))
	# 11返回结果：返回值：目前返回的是 None，可以根据需要修改为返回其他信息，例如最佳模型参数或评估结果。
	return res

# test 的函数，用于在测试集上进行预测。
# 该函数将测试数据分批次输入模型，获取模型的预测结果，并将所有批次的预测结果拼接成一个完整的预测矩阵。
# 参数说明：
# args：包含配置参数的对象，例如批量大小 batch_size。
# X_test：形状为 (N_test, d) 的测试集特征矩阵，其中 N_test 是测试样本数量，d 是特征维度。
# num_class：标签的数量（类别数）。
# model：训练好的神经网络模型，用于进行预测。
def test(args, X_test, num_class, model):
	# 2. 将测试数据转换为 PyTorch 张量并移动到 GPU
	X_test_tensor = torch.from_numpy(X_test).float().cuda().detach()
	# 3. 计算每个 epoch 的迭代次数
	# iter_per_epoch：计算每个 epoch 中的迭代次数，确保所有测试样本都被遍历到。使用 math.ceil 以处理最后一个不完整的批次。
	iter_per_epoch = int(math.ceil(X_test_tensor.shape[0] / args.batch_size))
	# 4. 初始化预测结果列表
	# Y_pred：一个空列表，用于存储每个批次的预测结果。最终会将这些批次的结果拼接成一个完整的预测矩阵。
	Y_pred = []
	# 5. 批量预测
	# torch.no_grad()：禁用梯度计算，因为测试过程中不需要反向传播梯度，这可以节省内存并加快计算速度。
	# for i in range(iter_per_epoch)：遍历每个批次，进行预测。
	# start_idx：计算当前批次的起始索引，确保从正确的样本开始取数据。
	# X_batch：从 X_test_tensor 中提取当前批次的特征数据。X_batch 的形状为 (batch_size, d)，其中 batch_size 是当前批次的样本数量（可能小于 args.batch_size，特别是在最后一个批次）。
	# Y_pred_batch = model.forward(X_batch)：将当前批次的特征数据输入模型，获取模型的预测结果。Y_pred_batch 是一个形状为 (batch_size, num_class) 的张量，表示每个样本的标签分布或概率。
	# Y_pred += [Y_pred_batch.detach().cpu().numpy()]：将当前批次的预测结果从 GPU 移回 CPU，并转换为 NumPy 数组，然后将其添加到 Y_pred 列表中。detach() 断开张量与计算图的连接，cpu() 将张量移回 CPU，numpy() 将张量转换为 NumPy 数组。
	with torch.no_grad():
		for i in range(iter_per_epoch):
			start_idx = (i * args.batch_size) % X_test_tensor.shape[0]
			X_batch = X_test_tensor[start_idx: start_idx + args.batch_size, :]
			Y_pred_batch = model.forward(X_batch)
			Y_pred += [Y_pred_batch.detach().cpu().numpy()]
	# 6. 拼接所有批次的预测结果
	# np.concatenate(Y_pred, axis=0)：将 Y_pred 列表中的所有批次预测结果沿着第 0 维（即样本维度）拼接成一个完整的预测矩阵。
	# 最终的 Y_pred 形状为 (N_test, num_class)，表示每个测试样本的标签分布或概率。
	Y_pred = np.concatenate(Y_pred, axis=0) 
	# 7. 返回预测结果
	# 输出：返回拼接后的预测矩阵 Y_pred，它是一个形状为 (N_test, num_class) 的 NumPy 数组，表示每个测试样本的标签分布或概率。
	return Y_pred





# a. 深度流行学习-准备过程[模型真实使用算法]
def deepm_train_global_prepare(args, X_train, Y_P_train):


	# 5. 初始化标签预测和标签传播的标签分布
	# Y_pred_np：初始化为 Y_P_train，表示当前的标签预测。
	# Y_lp_np：初始化为 Y_P_train，表示标签传播后的标签分布。
	# Y_pred_np = Y_P_train
	# Y_lp_np = Y_P_train

	# 6. 构建图结构和估计标签相关性矩阵
	# Wn：通过 build_graph 函数构建归一化的邻接矩阵，表示图中节点之间的相似度。
	# 检查 X_train 是否包含 NaN
	if np.isnan(X_train).any():
		print("警告：X_train 中包含 NaN 值！")
	Wn = build_graph(X_train, k=args.neighbors_num, args=args)
	# 估计标签相关性矩阵L：通过 estimating_label_correlation_matrix 函数估计标签相关性矩阵，表示不同标签之间的关联性。【这个貌似要去掉】
	L = estimating_label_correlation_matrix(Y_P_train)  # np.zeros(Y_P_train.shape) #

	# 标签传播：如果 args.using_lp 为 True，则调用 label_propagation 函数执行标签传播算法，更新标签分布 Y_lp_np。
	if args.using_lp:
		maxiter = args.maxiter
		# Y_lp_np = label_propagation(args, Wn, L, Y_pred_np, Y_P_train, Y_lp_np
		# 							, alpha=args.alpha, eta=args.eta, beta=args.beta,
		# 							maxiter=maxiter)  # 标签更新了，也就是Z更新了
		Y_recon,X_recon = label_propagation_with_x_reconstruction(args, Wn, L, Y_P_train, Y_P_train, None
									, alpha=args.alpha, eta=args.eta, beta=args.beta,
									maxiter=maxiter,X_train=X_train)  # 标签更新了，也就是Z更新了
	return Y_P_train, Y_recon, Wn, L,X_recon

# a. 深度流行学习-单次迭代
def deepm_train_iter(args, model_deepm, X_train_iter, Y_train_P_iter, Y_train_lp_iter):
	# 清除梯度：optimizer.zero_grad()，防止梯度累积。
	#optimizer.zero_grad()

	# # 计算索引位置和批次内的索引序号
	# start_idx = (i * batch_size) % X_train.shape[0]
	# idx = train_indicies[start_idx: start_idx + batch_size]
	# idx = train_indices # without batch

	# 获取当前批次的数据：根据索引 idx 获取当前批次的特征 X 和标签 Y_P 或 Y_lp。
	# get minibatch indices
	# randomly select a mini-batch of data
	# X = torch.from_numpy(X_train[idx, :]).float().detach().cuda()
	# Y_P = torch.from_numpy(Y_P_train[idx, :]).float().detach().cuda()
	# Y_lp = torch.from_numpy(Y_lp_np[idx, :]).float().detach().cuda()
	#X_train_iter = torch.from_numpy(X_train_iter).float().detach().cuda()
	Y_train_P_iter = torch.from_numpy(Y_train_P_iter).float().detach().cuda()
	Y_train_lp_iter = torch.from_numpy(Y_train_lp_iter).float().detach().cuda()
	X_train_iter.requires_grad = False
	Y_train_P_iter.requires_grad = False
	Y_train_lp_iter.requires_grad = False

	# 前向传播：将特征 X 输入模型，获取预测结果 Y_pred。
	Y_pred_iter,A = model_deepm.forward(X_train_iter,args)
	# 调整 Y_pred_iter 的形状
	Y_pred_iter = Y_pred_iter.squeeze(0)  # 将形状从 [1, 7] 调整为 [7]

	# 计算损失：根据是否使用标签传播，选择不同的损失函数：
	# 如果使用标签传播，使用均方误差损失 F.mse_loss(Y_pred, Y_lp)。
	# 否则，使用二元交叉熵损失 F.binary_cross_entropy(Y_pred, Y_P)。
	# 检查 Y_pred_iter 的类型
	#print("Y_pred_iter.shape:",Y_pred_iter.size,"Y_train_lp_iter.shape:",Y_train_lp_iter.shape)
	if args.using_lp:
		deepm_loss = F.mse_loss(Y_pred_iter, Y_train_lp_iter)
	else:
		deepm_loss = F.binary_cross_entropy(Y_pred_iter, Y_train_P_iter)

	# # 反向传播：loss.backward()计算梯度。
	# loss.backward()
	# # 更新参数：optimizer.step() 根据计算的梯度更新模型参数。
	# optimizer.step()
	return deepm_loss


# 0 深度运行-模型训练与评估(要被融合和改写的部分)
# 输入：
# args：包含配置参数的对象，例如学习率、批量大小、是否使用标签传播等。
# data：一个包含训练数据和测试数据的元组 (X_train, Y_train, Y_P_train, X_test, Y_test)。
# model：要训练的神经网络模型。
# optimizer：优化器，用于更新模型参数。
# training_now：布尔值，表示是否正在进行训练（默认为 True）。
# eval_every：每隔多少个 epoch 进行一次评估，默认为 5。
def deepm_run(args, data, model_deepm, optimizer, training_now=True, eval_every=5):
	res = None  # 定义返回值 define the return value
	# res_log 和 overall_loss_log：用于记录训练过程中的日志信息，分别记录详细的损失和每个epoch的总体损失。
	res_log = np.zeros(args.epochs * 150)
	overall_loss_log = np.zeros(args.epochs)

	# 3.解包数据
	# X_train：训练集的特征矩阵。
	# Y_train：训练集的真实标签（可能未使用）。
	# Y_P_train：训练集的初始标签分布（部分标记数据）。
	# X_test：测试集的特征矩阵。
	# Y_test：测试集的真实标签。
	# batch_size：每次训练时使用的批量大小。
	# num_class：标签的数量。
	X_train, Y_train, Y_P_train, X_test, Y_test = data
	# 4. 计算每个 epoch 的迭代次数
	batch_size = args.batch_size
	num_class = Y_P_train.shape[1]
	# loss初始值
	loss_pred = 0

	# number of iteration times per epoch
	# iter_per_epoch：每个 epoch 中的迭代次数，确保所有训练样本都被遍历到。使用 math.ceil 以处理最后一个不完整的批次。
	iter_per_epoch = int(math.ceil(X_train.shape[0] / batch_size))
	# iter_per_epoch = 1 # without batch


	# counter
	iter_cnt = 0

	Y_pred_np, Y_lp_np, Wn, L = deepm_train_global_prepare(args, X_train, Y_P_train)

	best_score = 0

	# 7. 训练循环
	# 打乱训练数据的索引 shuffle indices
	train_indicies = np.arange(X_train.shape[0])
	np.random.shuffle(train_indicies)

	# 8. 批量训练
	# keep track of losses
	# 设置模型为训练模式：model.train()。
	model_deepm.train()

	for i in range(iter_per_epoch):
		# 清除梯度：optimizer.zero_grad()，防止梯度累积。
		optimizer.zero_grad()

		# 计算索引位置和批次内的索引序号
		start_idx = (i * batch_size) % X_train.shape[0]
		idx = train_indicies[start_idx: start_idx + batch_size]
		# idx = train_indices # without batch

		# 获取当前批次的数据：根据索引 idx 获取当前批次的特征 X 和标签 Y_P 或 Y_lp。
		# get minibatch indices
		# randomly select a mini-batch of data
		X_train_iter = torch.from_numpy(X_train[idx, :]).float().detach().cuda()
		Y_train_P_iter = torch.from_numpy(Y_P_train[idx, :]).float().detach().cuda()
		Y_train_lp_iter = torch.from_numpy(Y_lp_np[idx, :]).float().detach().cuda()
		X_train_iter.requires_grad = False
		Y_train_P_iter.requires_grad = False
		Y_train_lp_iter.requires_grad = False

		deepm_loss = deepm_train_iter(args, model_deepm, X_train_iter, Y_train_P_iter, Y_train_lp_iter)

		deepm_loss.backward()
		# 更新参数：optimizer.step() 根据计算的梯度更新模型参数。
		optimizer.step()
	# approximating candidate labels

	# 9. 更新训练集X_train的标签预测标签Y_pred_np
	# 在每个epoch结束后，使用test函数对训练集进行预测，更新 Y_pred_np。
	Y_pred_np = test(args, X_train, num_class, model_deepm)

	# 10. 评估模型
	Z = Y_lp_np
	# 设置模型为评估模式：model.eval()。
	model_deepm.eval()

	# testing
	if args.no_verbose:
		eval_every = 1

	# 测试模型：每隔 eval_every 个 epoch，在测试集上进行预测，并调用 evaluate 函数评估模型性能，
	# 计算 r_loss（排名损失）、h_loss（汉明损失）和 ap（平均精度）。
	Y_test_pred = test(args, X_test, num_class, model_deepm)
	r_loss, h_loss, ap = evaluate(Y_test, Y_test_pred, threshold=args.threshold)

	# 打印日志：如果 args.no_verbose 为 False，则打印当前 epoch 的评估结果。
	if not args.no_verbose:
		print(' r_loss %.4f, h_loss %.4f, ap %.4f'
			  % (r_loss, h_loss, ap))
	# 11返回结果：返回值：目前返回的是 None，可以根据需要修改为返回其他信息，例如最佳模型参数或评估结果。
	return Y_pred_np, Y_test_pred, r_loss, h_loss, ap




# Python 脚本的主程序部分，使用 argparse 模块来解析命令行参数，并根据这些参数配置和运行一个深度学习模型。
# 该脚本主要用于训练和评估一个神经网络模型，特别适用于多标签分类任务，并结合了标签传播算法（Label Propagation）。
if __name__ == '__main__':
	# argparse.ArgumentParser：创建一个参数解析器，用于处理命令行参数。
	parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
	# Optimization options
	# --epochs：总训练轮数，默认为 300。
	# --batch-size：训练时的批量大小，默认为 32。
	# --lr 或 --learning-rate：初始学习率，默认为 0.01。
	# --momentum：SGD 优化器的动量，默认为 0.9。
	# --weight_decay：权重衰减（L2 正则化）系数，默认为 5e-5。
	# --neighbors_num：构建图时使用的 k 近邻数，默认为 10。
	# --threshold：评估时的阈值，默认为 0.7。
	# --hidden_size：隐藏层的神经元数量，默认为 '64,64'，表示两个隐藏层，每层 64 个神经元。
	# --gpuid：使用的 GPU ID，默认为 0。
	parser.add_argument('--epochs', default=300, type=int, metavar='N',
						help='number of total epochs to run')
	parser.add_argument('--batch-size', default=32, type=int, metavar='N',
						help='train batchsize')
	parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
						metavar='LR', help='initial learning rate')
	parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
						help='SGD momentum (default: 0.9)')
	parser.add_argument('--using_lp', action='store_true')
	parser.add_argument('--using_prep', action='store_true')
	parser.add_argument('--class_cons', type=float, default=0.0)
	parser.add_argument('--weight_decay', type=float, default=5e-5)
	parser.add_argument('--neighbors_num', type=int, default=10)
	parser.add_argument('--threshold', type=float, default=0.7)
	parser.add_argument('--hidden_size', type=str, default='64,64')
	parser.add_argument('--gpuid', type=int, default=0)

	# main parameters
	# --alpha：标签传播中基于图结构传播项的权重，默认为 0.01。
	# --eta：标签传播中基于初始标签分布约束项的权重，默认为 1。
	# --beta：标签传播中基于标签相关性约束项的权重，默认为 0.01。
	# --maxiter：标签传播的最大迭代次数，默认为 200。
	# --gamma：标签传播的学习率，默认为 0.01。
	# --tr_rate：训练集与测试集的比例，默认为 0.9。
	# --no-verbose：是否禁用日志输出，默认为 False。
	# --mode：模式选择，默认为 0。
	# --eval_every：每隔多少个 epoch 进行一次评估，默认为 5。
	# --using_lp：是否使用标签传播算法，默认为 False。
	# --using_prep：是否使用预处理数据，默认为 False。
	parser.add_argument('--alpha', type=float, default=0.01)
	parser.add_argument('--eta', type=float, default=1)
	parser.add_argument('--beta', type=float, default=0.01)

	parser.add_argument('--maxiter', type=int, default=200)
	parser.add_argument('--gamma', type=float, default=0.01)

	parser.add_argument('--tr_rate', type=float, default=0.9)

	parser.add_argument('--no-verbose', action='store_true')

	parser.add_argument('--mode', type=int, default=0)
	parser.add_argument('--eval_every', type=int, default=5)
	# parser.parse_args()：解析命令行参数，并将它们存储在 args 对象中。
	args = parser.parse_args()
    # 配置GPU
	# os.environ["CUDA_VISIBLE_DEVICES"]：设置可见的 GPU 设备，确保脚本只使用指定的 GPU。args.gpuid 是用户通过命令行传递的 GPU ID。
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)

	# 3. 处理隐藏层参数
	# args.hidden_size.split(',')：将字符串形式的隐藏层参数（如 '64,64'）按逗号分割成列表。
	# [int(n) for n in ...]：将每个元素转换为整数，最终得到一个包含隐藏层神经元数量的列表。
	args.hidden_size = [int(n) for n in args.hidden_size.split(',')]

	target = 'music_emotion'
	# 4. 打印配置信息
	# target：数据集名称，这里固定为 'music_emotion'。
	# print：打印数据集名称、学习率和是否使用标签传播的信息，便于用户确认配置。
	print('Dataset: {}, lr: {}, using_lp: {}'.format(target, args.lr, args.using_lp))

	# 5. 加载数据
	# file_name：数据文件的路径，格式为 'data/music_emotion'。
	# if args.using_prep：如果用户指定了 --using_prep，则加载预处理后的数据；否则，加载原始数据并进行划分。
	# load_preprocessed_data 和 load_data：这两个函数分别用于加载预处理数据和原始数据。load_data 函数还会根据 args.tr_rate 将数据划分为训练集和测试集。
	#file_name = 'data/' + target
	file_name = '/home/roger/codePlace/pythonCode/AIIA/pml_transformer/core/DEMIPLPLUS/graph/'+'data/' + target

	# 6. 解包数据
	# data：包含训练集特征、训练集标签、训练集初始标签分布、测试集特征和测试集标签的元组。
	# X_test, Y_test：从 data 中解包出测试集的特征和标签。
	if args.using_prep:
		print('Using preprocessed data ...')
		data = load_preprocessed_data(file_name)
	else:
		print('Using raw data ...')
		data = load_data(file_name, tr_rate=args.tr_rate)
	_, _, _, X_test, Y_test = data
	# 7. 初始化模型
	# DeepNet：自定义的神经网络模型类，接受输入特征维度、输出标签维度和隐藏层神经元数量作为参数。
	# X_test.shape[1]：测试集特征的维度，即输入层的神经元数量。
	# Y_test.shape[1]：测试集标签的维度，即输出层的神经元数量（标签数量）。
	# args.hidden_size：隐藏层的神经元数量列表。
	# .cuda()：将模型移动到 GPU 上进行加速计算。
	model = DeepNet(X_test.shape[1], Y_test.shape[1], args.hidden_size).cuda()
	# 8. 初始化优化器
	# torch.optim.SGD：使用随机梯度下降（SGD）优化器。
	# model.parameters()：获取模型的所有可训练参数。
	# lr=args.lr：设置学习率为 args.lr。
	# momentum=args.momentum：设置动量为 args.momentum。
	# weight_decay=args.weight_decay：设置权重衰减（L2 正则化）系数为 args.weight_decay。
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

	print(' ')
	# 9. 训练和评估模型
	# run_model：调用 run_model 函数，传入配置参数、数据、模型、优化器以及评估间隔 eval_every，开始训练和评估模型。
	#run_model(args, data, model, optimizer, eval_every=args.eval_every)
	deepm_run(args, data, model, optimizer, eval_every=args.eval_every)