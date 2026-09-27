import numpy as np
from sklearn.metrics import label_ranking_loss, hamming_loss, average_precision_score

# metric 的函数，用于计算多标签分类任务中的三种评估指标：
# 排名损失（Ranking Loss）、汉明损失（Hamming Loss） 和 平均精度（Average Precision, AP）。
# 这些指标可以帮助我们评估模型在多标签分类任务中的性能。
# 输入：
# y：形状为 (N, num_class) 的真实标签矩阵，其中 N 是样本数量，num_class 是标签的数量。每个元素是一个二进制值（0 或 1），表示该样本是否属于某个标签。
# y_pred：形状为 (N, num_class) 的预测标签矩阵，通常是由模型输出的概率值经过阈值处理后得到的二进制预测结果。
# scores：形状为 (N, num_class) 的预测分数矩阵，表示模型对每个样本和每个标签的置信度（通常是概率值）。scores 用于计算排名损失和平均精度。
def metric(y, y_pred, scores):
    # 2. 计算样本数量
    # n：样本数量，y.shape[0] 表示 y 的行数，即样本数量。1.0 * 将其转换为浮点数，确保后续的除法操作是浮点数除法。
    n = 1.0 * y.shape[0]
    # 3. 计算排名损失（Ranking Loss）
    # label_ranking_loss：这是一个 Scikit-learn 提供的函数，用于计算排名损失。
    # 排名损失衡量的是模型对正标签和负标签的排序能力。具体来说，它计算的是所有正标签的得分低于负标签的得分的比例。
    # 排名损失越低，说明模型对正标签的打分越高，对负标签的打分越低，模型的排序能力越好。
    r_loss = label_ranking_loss(y, scores)
    # 4. 计算汉明损失（Hamming Loss）
    # hamming_loss：这也是一个 Scikit-learn 提供的函数，用于计算汉明损失。
    # 汉明损失衡量的是预测标签与真实标签之间的差异比例。
    # 具体来说，它是每个样本中预测错误的标签数量占总标签数量的比例。
    # 汉明损失的取值范围是 [0, 1]，0 表示完全正确，1 表示完全错误。
    h_loss = hamming_loss(y, y_pred)
    ap = 0
    # 5. 计算平均精度（Average Precision, AP）
    # average_precision_score：这是 Scikit-learn 提供的函数，用于计算单个样本的平均精度。平均精度衡量的是模型在不同阈值下的精度-召回率曲线下的面积。对于每个样本，y[i] 是该样本的真实标签，scores[i] 是该样本的预测分数。average_precision_score 返回的是该样本的平均精度。
    # for i in range(y.shape[0])：遍历每个样本，计算每个样本的平均精度，并将其累加到 ap 中。
    # ap /= n：将所有样本的平均精度求平均，得到最终的平均精度。n 是样本数量，ap 是所有样本的平均精度之和。
    for i in range(y.shape[0]):
        ap += average_precision_score(y[i], scores[i])

    ap /= n
    # 6. 结果返回
    # 输出：返回三个评估指标：
    # r_loss：排名损失。
    # h_loss：汉明损失。
    # ap：平均精度。
    return r_loss, h_loss, ap

if __name__ == '__main__':
    pass