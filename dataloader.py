# 多实例学习（MIL, Multiple Instance Learning）任务的数据处理和模型准备的Python脚本
# 这一行是Shebang行，它告诉操作系统使用Python解释器来执行这个脚本。env命令会根据环境变量查找Python解释器的位置，这样即使Python安装在不同的路径下也能正确执行23。
#!/usr/bin/env python
# 指定文件编码为UTF-8，确保文件中包含的非ASCII字符能够被正确读取和显示。对于Python 3.x版本，默认编码已经是UTF-8，但在某些情况下，特别是为了兼容Python 2.x或明确指定编码时，这一行仍然是有用的。
# -*- coding: UTF-8 -*-

# import numpy as np：导入NumPy库，并将其命名为np，用于数值计算。NumPy是Python在科学计算领域取得成功的关键之一，提供了强大的数组操作功能
import numpy as np
# 导入Python的随机数模块，用于生成随机数或打乱列表等操作。这在数据预处理和模型训练过程中非常有用，例如打乱训练数据以提高模型的泛化能力
import random
# 导入PyTorch库，用于深度学习模型的构建和训练。PyTorch是一个流行的机器学习框架，支持动态计算图和自动微分，非常适合快速原型设计和研究
import torch
# 从SciPy库中导入io模块，专门用于读取MATLAB格式的数据文件（.mat）。SciPy是一个用于科学计算的Python库，提供了许多高效的数学算法和数据结构
import scipy.io as io
# 从PyTorch中导入数据处理工具，如Dataset和DataLoader，方便构建自定义数据集和数据加载器。
# 这些工具可以帮助我们更高效地管理数据流，特别是在处理大型数据集时
import torch.utils.data as data_utils

# 3. 定义辅助函数：该函数的作用是将给定的标签转换为one-hot编码形式。
# def to_categorical(y, nr_class):：定义一个名为to_categorical的函数，
# 接收两个参数：y（标签列表）和nr_class（类别数量）。
def to_categorical(y, nr_class):
    '''
    generate one-hot label 说明该函数用于生成one-hot编码的标签。One-hot编码是一种常用的表示分类标签的方法，其中每个类别用一个独热向量表示。
    '''
    # y_list = [0] * nr_class：初始化一个长度为nr_class的全零列表，表示每个类别的初始状态。这个列表将用于存储one-hot编码的结果。
    y_list = [0] * nr_class
    # 遍历输入的标签列表y。对于每个标签i，将其对应位置的值设为1，表示该类别存在。
    for i in y:
        # y_list[i] = 1：将y_list中索引为i的位置设为1，表示第i个类别存在。
        y_list[i] = 1
    # y_cate = np.array(y_list)：将列表y_list转换为NumPy数组，便于后续处理。NumPy数组提供了更多的操作和优化，适合进行数值计算。
    y_cate = np.array(y_list)
    # return y_cate：返回转换后的one-hot编码标签。这个标签可以用于模型训练和评估，特别是在分类任务中
    return y_cate

# 3.2 load_data_mat 该函数的主要作用是从MAT文件中加载数据集，并进行必要的预处理。
# def load_data_mat(mat_path, nr_fea, nr_class, normalize=False):：定义一个名为load_data_mat的函数，
# 接收四个参数：mat_path（MAT文件路径）、nr_fea（特征数量）、nr_class（类别数量）和normalize（是否归一化，默认为False）。
def load_data_mat(mat_path, nr_fea, nr_class, normalize=False):
    '''
    load the dataset in mat format 函数注释，说明该函数用于加载MAT格式的数据集。MAT文件通常用于存储矩阵数据，常用于科学计算和工程应用
    '''
    # 使用scipy.io.loadmat函数读取MAT文件，返回一个字典，其中包含了文件中的所有变量。mat_path指定了MAT文件的路径。
    data_mat = io.loadmat(mat_path)
    # data = data_mat['data']：【包id,对应的实例】从加载的MAT文件中提取名为data的变量，通常是一个包含多个实例的矩阵。每个实例可能包含多个特征和标签。
    data = data_mat['data']
    # all_ins_fea = []：初始化一个空列表all_ins_fea，用于存储所有实例的特征。
    all_ins_fea = []
    # all_ins_fea_tmp 初始化一个形状为(0, nr_fea)的空NumPy数组all_ins_fea_tmp，用于临时存储每个包中的实例特征。
    all_ins_fea_tmp = np.empty((0, nr_fea))
    # ins_num, bag_idx_of_ins = [], []：初始化两个空列表，（1）ins_num用于存储每个包中实例的数量。（2）bag_idx_of_ins每个实例所属的包索引。
    ins_num, bag_idx_of_ins = [], []
    # 初始化两个空列表:（1）bag_lab用于存储每个包的真实标签和（2）dummy_ins_lab为每个实例的虚拟标签。
    bag_lab, dummy_ins_lab = [], []
    # 初始化一个空列表partial_bag_lab，用于存储每个包的部分标签。
    partial_bag_lab = []
    # 初始化一个形状为(0, nr_class)的空NumPy数组partial_dummy_ins_lab，用于存储每个实例的部分标签。
    partial_dummy_ins_lab = np.empty((0, nr_class))
    # 初始化一个形状为(0, nr_class)的空NumPy数组partial_dummy_ins_lab_processed，用于存储处理后的部分标签。
    partial_dummy_ins_lab_processed = np.empty((0, nr_class))
    # 初始化两个形状为(0, nr_class)的空NumPy数组partial_bag_lab和partial_bag_lab_processed，分别用于存储每个包的部分标签及其处理后的版本。
    partial_bag_lab, partial_bag_lab_processed = np.empty((0, nr_class)), np.empty((0, nr_class))
    # bag_cnt = 1：初始化一个计数器bag_cnt，用于记录当前处理的包索引，从1开始编号。
    bag_cnt = 1
    # for i in range(data.shape[0]):：遍历data中的每一个包。data.shape[0]表示数据集中包的数量。
    for i in range(data.shape[0]):
        # 将当前包中的所有实例特征垂直堆叠到all_ins_fea中。data[i, 0]表示第i个包的所有实例特征。
        # 【1.all_ins_fea所有样本实例样本数*特征数。data[i, 0]中0里面放的是包内所有实例样本，其中all_ins_fea_tmp是上一轮所有的堆起数据】
        all_ins_fea = np.vstack((all_ins_fea_tmp, data[i, 0]))
        # 更新all_ins_fea_tmp，使其包含最新的实例特征。
        all_ins_fea_tmp = all_ins_fea
        # ins_num_tmp为当前包中实例的数量。
        ins_num_tmp = data[i, 0].shape[0]
        # 【2.ins_num列表中存放当前包中实例的数量】。
        ins_num.append(ins_num_tmp)
        # 创建一个长度为ins_num_tmp的列表bag_idx_of_ins_tmp，所有元素都为当前包的索引bag_cnt
        # bag_idx_of_ins_tmp 创建了一个列表 bag_idx_of_ins_tmp，其中的每个元素都是 bag_cnt，并且这个列表的长度是 ins_num_tmp。
        # bag_cnt：这是一个值，可以是任何数据类型，但在上下文中很可能是一个整数，表示某个特定的“bag”的索引或者计数。
        # [bag_cnt]：这是一个包含单个元素 bag_cnt 的列表。
        # * ins_num_tmp：这是Python中的乘法操作，用于复制列表。它将列表 [bag_cnt] 复制 ins_num_tmp 次。
        # bag_idx_of_ins_tmp = [bag_cnt] * ins_num_tmp：最终，这个语句将复制后的列表赋值给变量 bag_idx_of_ins_tmp。
        # 例如，如果 bag_cnt 是 3 并且 ins_num_tmp 是 5，那么执行这个语句后，bag_idx_of_ins_tmp 将会是 [3, 3, 3, 3, 3]，一个包含五个元素的列表，每个元素都是 3。
        bag_idx_of_ins_tmp = [bag_cnt] * ins_num_tmp
        # 【3.bag_idx_of_ins记录所有实例的包索引号】将bag_idx_of_ins_tmp列表拼接到bag_idx_of_ins列表后面，记录每个实例所属的包索引，形成一个长序列。
        bag_idx_of_ins = bag_idx_of_ins + bag_idx_of_ins_tmp
        # 增加包计数器bag_cnt，准备处理下一个包。
        bag_cnt += 1
        # the ground-truth labels of bags 处理真实标签
        # 从data中提取第i个包的真实标签，并将其转换为列表bag_lab_tmp。data[i, 2]表示第i个包的真实标签矩阵，
        # flatten()将其展平为一维数组，-1是为了将标签从1-based转换为0-based。【matlab是1开始编号，python，java数组是0开始编号】
        bag_lab_tmp = list(data[i, 2].flatten() - 1)
        # 【4.bag_lab总的包真实标签列表，是一维平铺的】，将当前包的真实标签bag_lab_tmp拼接进来。
        bag_lab = bag_lab + bag_lab_tmp
        # dummy_ins_lab_tmp【列表中嵌列表，长度ins_num_tmp】，所有元素都为当前包的真实标签bag_lab_tmp。
        dummy_ins_lab_tmp = [bag_lab_tmp] * ins_num_tmp
        # 【5.dummy_ins_lab 所有样本实例对应的真实标签集合（列表内嵌套列表）】将dummy_ins_lab_tmp添加到dummy_ins_lab中，记录每个实例的虚拟标签。
        dummy_ins_lab = dummy_ins_lab + dummy_ins_lab_tmp
        # 【6.partial_bag_lab=包的偏标签】。the partial labels of bags
        # 从data中提取第i个包的偏标签，并将其转换为列表partial_bag_lab_tmp。
        # data[i, 1]表示第i个包的部分标签矩阵，flatten()将其展平为一维数组，-1是为了将标签从1-based转换为0-based。
        partial_bag_lab_tmp = list(data[i, 1].flatten() - 1)
        # 将偏标签转换为one-hot编码形式，调用to_categorical函数。
        partial_bag_lab_tmp = to_categorical(partial_bag_lab_tmp, nr_class)
        # 将partial_bag_lab_tmp扩展为二维数组，以便后续堆叠操作。
        partial_bag_lab_tmp = np.expand_dims(partial_bag_lab_tmp, axis=0)
        # partial_bag_lab，将当前包的偏标签垂直堆叠到partial_bag_lab。
        partial_bag_lab = np.vstack((partial_bag_lab, partial_bag_lab_tmp))
        # 【7.partial_dummy_ins_lab 每个样本实例对应的偏标签】将当前包的部分标签重复ins_num_tmp次，生成每个实例的部分标签。
        partial_dummy_ins_lab_tmp = partial_bag_lab_tmp.repeat(ins_num_tmp, axis=0)
        # 将当前包的实例部分标签添加到partial_dummy_ins_lab中，垂直堆叠。
        partial_dummy_ins_lab = np.vstack((partial_dummy_ins_lab, partial_dummy_ins_lab_tmp))

    # 将bag_idx_of_ins转换为NumPy数组。
    bag_idx_of_ins = np.array(bag_idx_of_ins)
    # 将bag_idx_of_ins扩展为二维数组，即纵向的，每行表示一个实例的包索引。
    bag_idx_of_ins = np.expand_dims(bag_idx_of_ins, axis=1)
    # 将bag_lab转换为NumPy数组，便于后续处理。
    bag_lab = np.array(bag_lab)
    # 将dummy_ins_lab转换为NumPy数组，便于后续处理。
    dummy_ins_lab = np.array(dummy_ins_lab)
    # 【lab_inx_fea，包含每个实例的(真实标签，包索引，实例的特征)，水平扩展开】将虚拟标签、包索引和实例特征水平拼接在一起，形成一个新的数组lab_inx_fea，每个实例的信息都包含在这行中。
    lab_inx_fea = np.hstack((dummy_ins_lab, bag_idx_of_ins, all_ins_fea))
    # 计算每个实例的偏标签总数，并将其扩展为二维数组 [sum,1],每行代表1个样本，每行里的值代表偏标签数量。
    nr_partial_lab_per_ins = np.expand_dims(np.sum(partial_dummy_ins_lab, 1), axis=1)
    # 对每个实例的偏标签进行归一化处理，值为1/nr_partial_lab_per_ins 或0,使其总和为1。
    partial_dummy_ins_lab_processed = partial_dummy_ins_lab / nr_partial_lab_per_ins
    # nr_partial_lab_per_bag=计算每个包的部分标签总数，并将其扩展为二维数组。
    nr_partial_lab_per_bag = np.expand_dims(np.sum(partial_bag_lab, 1), axis=1)
    # 对每个包的部分标签进行归一化处理，使其总和为1。
    partial_bag_lab_processed = partial_bag_lab / nr_partial_lab_per_bag

    # 如果normalize参数为True，则对特征进行归一化处理。
    if normalize:
        # 计算所有实例特征的均值和标准差。
        data_mean, data_std = np.mean(all_ins_fea, 0), np.std(all_ins_fea, 0)
        # 计算所有实例特征的最小值和最大值。
        data_min, data_max = np.min(all_ins_fea, 0), np.max(all_ins_fea, 0)
        # 对特征进行标准化处理，使其均值为0，标准差为1。
        all_ins_fea_norm = (all_ins_fea - data_mean) / data_std
        # 使用标准化后的特征。
        all_ins_fea = all_ins_fea_norm

    # 将all_ins_fea转换为PyTorch张量，便于后续的模型训练。
    all_ins_fea = torch.from_numpy(all_ins_fea)
    # 将bag_idx_of_ins转换为PyTorch张量。
    bag_idx_of_ins = torch.from_numpy(bag_idx_of_ins)
    # 将dummy_ins_lab转换为PyTorch张量。
    dummy_ins_lab = torch.from_numpy(dummy_ins_lab)
    # 将bag_lab转换为PyTorch张量。
    bag_lab = torch.from_numpy(bag_lab)
    # 将partial_bag_lab转换为PyTorch张量。
    partial_bag_lab = torch.from_numpy(partial_bag_lab)
    # 将partial_bag_lab_processed转换为PyTorch张量。
    partial_bag_lab_processed = torch.from_numpy(partial_bag_lab_processed)
    # 返回处理后的特征、包索引、虚拟标签、真实标签、偏标签及其处理后的版本。这些数据可以用于后续的模型训练和评估。
    return all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab, partial_bag_lab_processed

# 4. load_idx_mat的函数，接收一个参数idx_file，表示MAT文件的路径。
# 该函数用于加载MAT文件中的训练和测试索引。
def load_idx_mat(idx_file):
    '''
    函数用于加载MAT格式的索引文件。load the index in mat format
    '''
    # 使用scipy.io.loadmat函数读取MAT文件，返回一个字典，其中包含了文件中的所有变量。idx_file指定了MAT文件的路径。
    idx = io.loadmat(idx_file)
    # 从加载的MAT文件中提取名为trainIndex的变量，通常是一个包含训练索引的数组。
    idx_tr_np = idx['trainIndex']
    # 从加载的MAT文件中提取名为testIndex的变量，通常是一个包含测试索引的数组。
    idx_te_np = idx['testIndex']
    # 将trainIndex展平为一维数组，并转换为列表idx_tr。
    idx_tr = list(np.array(idx_tr_np).flatten())
    # 将testIndex展平为一维数组，并转换为列表idx_te。
    idx_te = list(np.array(idx_te_np).flatten())
    # 打乱idx_tr中的索引顺序，确保训练数据的随机性。
    random.shuffle(idx_tr)
    # 打乱idx_te中的索引顺序，确保测试数据的随机性。
    random.shuffle(idx_te)
    # 返回训练和测试索引列表。这些索引可以用于分割数据集，以便进行模型训练和评估
    return idx_tr, idx_te

# 4. 定义数据集类
# 定义一个名为MIPLDataset的类，继承自torch.utils.data.Dataset。这个类用于创建自定义的数据集，以便与PyTorch的DataLoader一起使用。
class MIPLDataset(data_utils.Dataset):
    # 定义类的构造函数，接收四个参数：bags_list（包列表）、partial_bag_lab（部分标签）、true_bag_lab（真实标签）和F（其他特征）。构造函数用于初始化数据集对象。
    def __init__(self, bags_list, partial_bag_lab, true_bag_lab, F):
        # 将bags_list赋值给类的成员变量self.bags_list，用于存储所有包的数据。
        self.bags_list = bags_list
        # 将partial_bag_lab赋值给类的成员变量self.partial_bag_lab，用于存储每个包的部分标签。
        self.partial_bag_lab = partial_bag_lab
        # 将true_bag_lab赋值给类的成员变量self.true_bag_lab，用于存储每个包的真实标签。
        self.true_bag_lab = true_bag_lab
        # 将F赋值给类的成员变量self.f，用于存储其他特征。
        self.f = F

    # 定义__len__方法，返回数据集的长度。这个方法用于告知DataLoader数据集中有多少个样本。
    def __len__(self):
        # 返回self.true_bag_lab的长度，即数据集中包的数量。
        return len(self.true_bag_lab)

    # 定义__getitem__方法，接收一个索引index，返回对应索引处的样本。这个方法用于按索引访问数据集中的样本。
    def __getitem__(self, index):
        # 获取索引为index的包。
        bag = self.bags_list[index]
        # 获取索引为index的包的偏标签。
        partial_bag_label = self.partial_bag_lab[index]
        # 获取索引为index的包的真实标签。
        true_bag_label = self.true_bag_lab[index]
        # 从self.f中获取索引为index的样本实例特征。
        i_f = self.f[index]
        # 返回包、偏标签、真实标签、样本特征以及索引。这些信息可以用于模型训练和评估。
        return bag, partial_bag_label, true_bag_label, i_f, index

# 定义一个名为create_bags的函数，
# 接收五个参数：all_ins_fea（所有实例的特征）、bag_idx_of_ins（每个实例所属的包索引）、
# dummy_ins_lab（每个实例的虚拟真实标签）、bag_lab（每个包的真实标签）、partial_bag_lab（每个包的偏标签）和idx_list（包索引列表）。
# 该函数用于根据给定的包索引列表创建包数据。
def create_bags(all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, 
                 partial_bag_lab, idx_list):
    # 初始化一个空列表bags_list，用于存储创建的包。
    bags_list = []
    # 初始化一个形状为(0,)的空张量ins_lab，用于存储所有实例的虚拟标签。
    ins_lab = torch.empty(0, dtype=torch.uint8)
    # 初始化一个形状为(0,)的空张量y_bag_partial，用于存储所有包的部分标签。
    y_bag_partial = torch.empty(0, dtype=torch.float64)
    # 初始化一个形状为(0,)的空张量true_bag_lab，用于存储所有包的真实标签。
    true_bag_lab = torch.empty(0, dtype=torch.uint8)
    # 遍历idx_list中的每个包索引i。
    for i in idx_list:
        # 创建一个布尔掩码,bag_idx_of_ins_a_bag表示哪些实例属于当前包i。
        bag_idx_of_ins_a_bag = bag_idx_of_ins == i
        # 去除bag_idx_of_ins_a_bag中的冗余维度，使其变为一维数组。
        bag_idx_of_ins_a_bag = np.squeeze(bag_idx_of_ins_a_bag)
        # 从all_ins_fea中提取属于当前包i的所有实例特征，形成一个新的包bag。
        bag = all_ins_fea[bag_idx_of_ins_a_bag, :]
        # 将当前包中所有实例的虚拟标签拼接到ins_lab中。squeeze(1)用于移除多余的维度。
        ins_lab = torch.cat((ins_lab, 
                             dummy_ins_lab[bag_idx_of_ins_a_bag].squeeze(1)
                             ), 0)
        # 将当前包的部分标签拼接到y_bag_partial中。unsqueeze(0)用于增加一个维度，以便后续拼接。
        y_bag_partial = torch.cat((y_bag_partial, 
                                   partial_bag_lab[i - 1, :].unsqueeze(0) 
                                   ),0)
        # 将当前包的真实标签拼接到true_bag_lab中。view(1, 1)用于调整标签的形状。
        true_bag_lab = torch.cat((true_bag_lab,
                                  bag_lab[i - 1].view(1, 1)
                                  ), 0)
        # 将当前包bag添加到bags_list中。
        bags_list.append(bag)
    # bags_list：包含所有创建的包的列表。 ins_lab：所有实例的虚拟标签。 y_bag_partial：所有包的部分标签。 true_bag_lab：所有包的真实标签。
    return bags_list, ins_lab, y_bag_partial, true_bag_lab

# mil_collate_fn 函数 ，自定义的collate_fn函数，用于在PyTorch的DataLoader中处理多实例学习任务的数据批次。
# batch：一个批次的数据，通常是由多个样本组成的列表，每个样本是一个元组，包含五个元素：包、部分标签、真实标签、其他特征和索引。
def mil_collate_fn(batch):
    # 提取每个样本中的包（item[0]），并将其存储在一个列表X中。由于每个包可能包含不同数量的实例，因此X是一个包含不同长度张量的列表。
    X = [item[0] for item in batch]
    # 提取每个样本的部分标签（item[1]），并使用torch.stack将其堆叠成一个张量S
    S = torch.stack([item[1] for item in batch])
    # 提取每个样本的真实标签（item[2]），并使用torch.stack将其堆叠成一个张量Y。
    Y = torch.stack([item[2] for item in batch])
    # 提取每个样本的特征（item[3]），并使用torch.stack将其堆叠成一个张量F。
    F = torch.stack([item[3] for item in batch])
    # 提取每个样本的索引（item[4]），并将其存储在一个列表index中。
    index = [item[4] for item in batch]
    # X：包含所有包的列表。
    # S：包含所有部分标签的张量。
    # Y：包含所有真实标签的张量。
    # F：包含所有其他特征的张量。
    # index：包含所有索引的列表。
    return X, S, Y, F, index

# setup_scatter 函数
# 将多个包的实例特征合并成一个大张量，并生成辅助信息，用于后续的散点图绘制或模型训练。
# Xs：[350-包的数量,48-包内实例数,128特征维度]一个包含多个包的列表，每个包是一个张量，表示该包内的所有实例特征。
def setup_scatter(Xs):
    # 获取第一个包所在的设备（CPU或GPU），以确保后续操作在同一设备上进行。
    device = Xs[0].device
    # 将所有包的实例特征沿着第0维（即实例维度）拼接成一个大张量x(包的这维去掉，拼到包内实例这一维度)，形状为(N, D)，其中N是所有包中实例的总数，D是特征维度。
    x = torch.cat(Xs, dim=0)
    # i=生成实例索引[每个实例对应的包索引号]
    # torch.full((x.shape[0],), idx)：为每个包中的实例生成一个与该包索引idx相同的张量，形状为(x.shape[0],)，其中x.shape[0]是该包中实例的数量。
    # [torch.full((x.shape[0],), idx) for idx, x in enumerate(Xs)]：对每个包执行上述操作，生成一个包含所有包实例索引的列表。
    # torch.cat(...).to(device)：将这些张量沿着第0维拼接成一个大张量i，并将其移动到指定的设备上。i的形状为(N,)，表示每个实例所属的包索引。
    i = torch.cat([torch.full((x.shape[0],), idx) for idx, x in enumerate(Xs)]).to(device)
    # i_ptr=生成包起始位置,[0,36,84...]共513个
    # i.bincount()：计算每个包中实例的数量，返回一个张量，形状为(M,)，其中M是包的数量。
    # i.bincount().cumsum(0)：对上述结果进行累积求和，得到每个包的起始位置。例如，如果第一个包有3个实例，第二个包有5个实例，则累积求和的结果为[3, 8]，表示第二个包的起始位置是第8个实例。
    # torch.cat([torch.tensor([0], device=device), ...])：在累积求和结果的前面添加一个0，表示第一个包的起始位置是第0个实例。最终，i_ptr的形状为(M+1,)，表示每个包的起始位置。
    i_ptr = torch.cat([torch.tensor([0], device=device), i.bincount().cumsum(0)]) # recording the start position of each bag
    # x：所有包中实例特征的拼接张量，形状为(N（包内的实例铺平）, 特征维度)。
    # i：每个实例所属的包索引，形状为(N,)。
    # i_ptr：每个包的起始位置，形状为(M+1,)，用于快速定位每个包的实例范围。
    return x, i, i_ptr