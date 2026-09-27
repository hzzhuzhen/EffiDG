# 多标签分类任务的数据加载、预处理和评估功能。
# 它使用了 scipy.io 来读取 .mat 文件格式的数据，并提供了多种数据分割方式（如训练/测试集划分和 k 折交叉验证）。
# 此外，代码还定义了评估模型性能的函数。
# numpy：用于数值计算。
# scipy.io：用于读取和写入 .mat 文件。
# metric：自定义的评估函数模块，包含多个评估指标的计算。
# sklearn.preprocessing：用于数据预处理，例如标准化。
# sklearn.metrics：提供了多种评估指标的计算函数。
import numpy as np
import scipy.io as sio
from metric import metric
from sklearn import preprocessing
import math
from sklearn.metrics import f1_score, label_ranking_loss, hamming_loss, coverage_error, average_precision_score

# 2. 标签转换函数：convert_labels
# 将标签矩阵 Y 中的所有非零值转换为 1，确保标签是二进制形式（0 或 1），适用于多标签分类任务。
def convert_labels(Y):
    Y = (Y > 0) * 1
    # convert to 0/1 labels
    return Y

# 3. 数据加载函数：load_data
# 功能：从 .mat 文件中加载数据，并将其划分为训练集和测试集。
# 输入：
# path：数据文件的路径。
# tr_rate：训练集占总数据的比例，默认为 0.8。
# observe_rate：观察率（未在函数中使用）。
# 逻辑：
# 读取数据：使用 sio.loadmat 读取 .mat 文件中的数据。
# 处理 NUS-WIDE 数据集：如果路径中包含 "nus"，则对数据进行转置（因为 NUS-WIDE 数据集的存储格式不同），并进行标准化。
# 处理其他数据集：对于其他数据集，直接读取数据并进行标准化。
# 打印数据大小：输出特征矩阵 X、真实标签 Y 和部分标签 Y_P 的形状。
# 随机打乱数据：生成一个随机排列 perm，并根据该排列重新排序 X、Y 和 Y_P。
# 划分训练集和测试集：根据 tr_rate 将数据划分为训练集和测试集。
# 输出：
# 返回一个元组 (trX, trY, trpY, tsX, tsY)，分别表示训练集特征、训练集标签、训练集部分标签、测试集特征和测试集标签。
def load_data(path, tr_rate=0.8, observe_rate=0.2):
    data = sio.loadmat(path)

    if path.find('nus') >= 0:
        X = data['data'].T
        X = preprocessing.scale(X)
        Y = data['target'].T
        Y_P = data['train_p_target'].T
    else:
        X = data['data']
        X = preprocessing.scale(X)
        # pre-processing
        Y = data['target'].T
        Y_P = data['partial_labels'].T
    print('Data size (X, Y, Y_P): ', X.shape, Y.shape, Y_P.shape)

    data_num = int(X.shape[0])
    perm = np.arange(data_num)
    np.random.shuffle(perm)
    X = X[perm]
    Y = Y[perm]
    Y_P = Y_P[perm]
    # 训练集测试集拆分
    part = int(data_num * tr_rate)
    trX = X[0:part, :]
    trY = Y[0:part, :]
    trpY = Y_P[0:part, :]
    tsX = X[part:data_num, :]
    tsY = Y[part:data_num, :]

    return (trX, trY, trpY, tsX, tsY)

# k 折交叉验证数据加载函数：load_data_k_fold
# 功能：从 .mat 文件中加载数据，并将其划分为 k 折，用于 k 折交叉验证。
# 输入：
# path：数据文件的路径。
# k：折数，默认为 10。
# 逻辑：
# 读取数据：与 load_data 类似，读取数据并进行标准化。
# 随机打乱数据：生成一个随机排列 perm，并根据该排列重新排序 X、Y 和 Y_P。
# 划分 k 折：将数据划分为 k 折，每折包含相同数量的样本（最后一折可能稍多一些）。
# 输出：
# 返回三个列表 (X_folds, Y_folds, YP_folds)，分别包含 k 个折叠的特征矩阵、标签矩阵和部分标签矩阵。

def load_data_k_fold(path, k=10):
    data = sio.loadmat(path)
    if path.find('nus') >= 0:
        X = data['data'].T # transpose X for NUSWIDE
        X = preprocessing.scale(X)
        Y = data['target'].T
        Y_P = data['train_p_target'].T
    else:
        X = data['data']
        X = preprocessing.scale(X)
        # pre-processing
        Y = data['target'].T
        Y_P = data['partial_labels'].T

    data_num = int(X.shape[0])
    perm = np.arange(data_num)
    np.random.shuffle(perm)
    X = X[perm]
    Y = Y[perm]
    Y_P = Y_P[perm]

    X_folds = [None] * k
    Y_folds = [None] * k
    YP_folds = [None] * k
    for i in range(k):
        start = int(data_num * (1.0 * i / k))
        if i < k - 1:
            end = int(data_num * (1.0 * (i+1) / k))
        else:
            end = data_num
        X_folds[i] = X[start:end, :]
        Y_folds[i] = Y[start:end, :]
        YP_folds[i] = Y_P[start:end, :]

    return (X_folds, Y_folds, YP_folds)

# 5. 模型评估函数：evaluate
# 功能：评估模型的性能，计算多个评估指标。
# 输入：
# Y_test：形状为 (N, num_class) 的真实标签矩阵。
# scores：形状为 (N, num_class) 的预测分数矩阵。
# threshold：用于将预测分数转换为二进制预测结果的阈值，默认为 0.5。
# 逻辑：
# 二值化预测结果：将预测分数 scores 转换为二进制预测结果 y_pred，即 scores > threshold 的位置设为 1，否则设为 0。
# 调用 metric 函数：使用自定义的 metric 函数计算排名损失、汉明损失和平均精度。
# 输出：
# 返回 metric 函数的输出，即排名损失、汉明损失和平均精度。
def evaluate(Y_test, scores, threshold=0.5):
    return metric(Y_test, 1 * (scores > threshold), scores)

# 6. 预处理数据加载函数：load_preprocessed_data
# 功能：加载已经预处理过的数据。
# 输入：
# path：数据文件的路径。
# 逻辑：
# 读取预处理数据：从 _preprocessed.mat 文件中读取预处理后的数据。
# 返回数据：返回训练集特征、训练集标签、训练集部分标签、测试集特征和测试集标签。
def load_preprocessed_data(path):
    data = sio.loadmat(path + '_preprocessed')
    return (data['trX'].copy(), data['trY'].copy(), data['trpY'].copy(), data['tsX'].copy(), data['tsY'].copy())

# 预处理测试代码：用于加载数据并保存预处理后的数据。
# 逻辑：
# 设置数据路径：path 设置为 'data/music_emotion'，表示数据文件的路径。
# 加载数据：调用 load_data 函数加载数据，并将其划分为训练集和测试集。
# 保存预处理数据：将加载的数据保存到 _preprocessed.mat 文件中，方便后续使用。
# pass：这是一个占位符，表示主程序暂时没有其他操作。
if __name__ == '__main__':
    path = 'data/music_emotion'
    trX, trY, trpY, tsX, tsY = load_data(path)
    sio.savemat(path + '_preprocessed.mat', {
        'trX': trX,
        'trY': trY,
        'trpY': trpY,
        'tsX': tsX,
        'tsY': tsY
    })
    pass