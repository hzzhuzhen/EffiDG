# os：用于文件路径的操作。
import os
# time：用于记录程序运行的时间。
import time
# torch：PyTorch深度学习框架的主要库21。
import torch
# numpy：用于数值计算的库。
import numpy as np
# DataLoader：PyTorch的数据加载器，用于批量处理数据集7。
from torch.utils.data import DataLoader
# accuracy_score：来自sklearn.metrics，用于计算分类准确率13。
from sklearn.metrics import accuracy_score
# device, logging, seed_everything, args：从utils模块导入，分别用于设备管理、日志记录、设置随机种子以确保结果可复现以及命令行参数解析。
from utils import device, logging, seed_everything, args
# EffiDG：自定义的多实例偏标签学习模型类。
#from model_graphV2 import EffiDG
from model_graphV3 import EffiDG
# load_data_mat, load_idx_mat, MIPLDataset, create_bags, mil_collate_fn：从dataloader模块导入，用于加载数据、创建数据集和自定义数据加载器。
from dataloader import load_data_mat, load_idx_mat, MIPLDataset, create_bags, mil_collate_fn
# import logging
# logging.basicConfig(level=logging.INFO)
# logging.info("测试日志")
# import logging
# root = logging.getLogger()
# print("Root logger level:", root.level)  # 30=WARNING, 20=INFO, 10=DEBUG
# print("Root logger handlers:", root.handlers)
# print("Effective level:", root.getEffectiveLevel())
# 2. 定义全局变量和辅助函数
# GLOBAL_WORKER_ID：全局变量，未在代码中使用，可能是为了后续扩展准备的。
GLOBAL_WORKER_ID = None

# adjust_alpha：定义了一个动态调整消歧权重的函数，随着训练轮次的增加，权重逐渐减小，这有助于在训练初期更加强调消歧的重要性。
# Weights of dynamic disambiguation
def adjust_alpha(epochs):
    alpha_list = [1.0] * epochs
    for ep in range(epochs):
        alpha_list[ep] = (epochs - ep) / (epochs)
    return alpha_list

# 3. 加载数据
# Load data
# 通过命令行参数构建数据路径和索引路径。
data_path = os.path.join(args.data_path, args.ds)
index_path = os.path.join(data_path, args.index)
# 构建MAT文件名，并记录日志。
# mat_name = args.ds + "_" + args.ds_suffix + ".mat"
# logging.info('MAT File Name: {}'.format(mat_name))
# mat_path = os.path.join(data_path, mat_name)
# ds_name = mat_name[0:-4]
mat_name = args.ds + '.mat'
if args.ds_suffix is not None:
    mat_name = args.ds + '_' + args.ds_suffix + '.mat'
logging.info('MAT File Name: {}'.format(mat_name))
mat_path = os.path.join(data_path, mat_name)
ds_name = mat_name[0:-4]

# 调用load_data_mat函数加载MAT文件中的数据，包括所有实例特征、包内实例索引、虚拟实例标签、包标签、部分包标签及其处理后的版本。
all_ins_fea, bag_idx_of_ins, dummy_ins_lab, bag_lab, partial_bag_lab, partial_bag_lab_processed = load_data_mat(
    mat_path, args.nr_fea, args.nr_class, normalize=args.normalize)

# 4. 设置随机种子
# Set the seed for reproducibility
# 创建一个PyTorch生成器并手动设置种子为0，确保数据加载过程中的一致性。
g = torch.Generator()
g.manual_seed(0)
# 定义worker_init_fn函数，用于初始化多线程数据加载器的工作进程，每个工作进程都会根据其ID设置不同的随机种子，以保证结果的可复现性。
def worker_init_fn(worker_id, seed_value = args.seed):
    seed_everything(seed_value + worker_id)

# 5. 主程序逻辑
if __name__ == "__main__":
    # 检查是否启用了CUDA支持，并记录相应的日志信息。
    if args.cuda:
        logging.info('\tGPU is available!')
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        logging.info('\tGPU is unavailable!')
        print("GPU is unavailable!")

    # 根据args.smoke_test参数决定是否只进行快速测试（仅使用一个折叠）或完整的交叉验证（使用多个折叠）。
    if args.smoke_test:
        all_folds = ['index1.mat']
    else:
        all_folds = ['index1.mat', 'index2.mat', 'index3.mat', 'index4.mat', 'index5.mat',
            'index6.mat', 'index7.mat', 'index8.mat', 'index9.mat', 'index10.mat']
    # 设置全局随机种子，确保整个实验过程的结果可复现。
    seed_everything(args.seed)
    num_fold = len(all_folds)
    # 记录开始时间，并初始化用于存储每次试验结果的数组accuracy。
    start_time = time.time()
    logging.info("\t================================ START ========================================")
    print("Start Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    # 最外层的重复试验次数（每次试验包含10次数据交叉）
    num_trial = args.nr_trial
    # 提前计算好学习率衰减
    alpha_list = adjust_alpha(args.epochs)
    # accuracy=记录多次试验、多次交叉数据验证的准确率结果矩阵
    accuracy = np.empty((num_trial, num_fold))

    # 6. 交叉校验循环
    for trial_i in range(num_trial):
        num_i = 0
        # 对于每个试验（trial_i），遍历所有的折叠（fold_i）。
        for fold_i in all_folds:
            logging.info('\t----------------time: {}, fold: {} ----------------'.format(trial_i, fold_i))
            print('\t----------------time: {}, fold: {} ----------------'.format(trial_i, fold_i))
            idx_file = os.path.join(index_path, fold_i)

            # Load the Index and Dataset
            # 1.加载当前折叠的训练和测试索引。
            idx_tr, idx_te = load_idx_mat(idx_file)

            # Generate Bags based on Index and Dataset
            # 2.使用create_bags函数根据索引生成训练和测试的包数据。
            # x_tr=bags_list：包含所有创建的包的列表。 y_ins_tr=ins_lab：所有实例的虚拟标签。
            # s_bag_tr=y_bag_partial：所有包的偏标签。 y_bag_tr=true_bag_lab：所有包的真实标签。
            x_tr, y_ins_tr, s_bag_tr, y_bag_tr = create_bags(all_ins_fea, bag_idx_of_ins, 
                                                            dummy_ins_lab, bag_lab, 
                                                            partial_bag_lab_processed, 
                                                            idx_tr)
            x_te, y_ins_te, s_bag_te, y_bag_te = create_bags(all_ins_fea, bag_idx_of_ins, 
                                                            dummy_ins_lab, bag_lab, 
                                                            partial_bag_lab_processed, 
                                                            idx_te)

            cov_b = torch.ones((len(x_tr), 1))

            # 3.初始化模型，并确保所有变量都在同一设备上（CPU或GPU），同时转换为float32类型以节省内存。
            # Initial Model 训练特征x，shape=(训练样本数，1)的全1覆盖，
            # x_tr训练特征，cov_b[训练数据长度]全1，s_bag_tr，包偏标签
            model = EffiDG.initialize_model(x_tr, cov_b, s_bag_tr, args)

            # Ensure Variablie in the Same Device and Convert to Float32
            model.to(device)
            x_tr, x_te, y_ins_tr, s_bag_tr, y_bag_tr = \
                [i.to(device).to(torch.float32) for i in x_tr], [i.to(device).to(torch.float32) for i in x_te], \
                y_ins_tr.to(device).to(torch.float32), s_bag_tr.to(device).to(torch.float32), y_bag_tr.to(device).to(torch.float32)
            cov_b = cov_b.to(device).to(torch.float32)
            
            # Generate DataLoader
            # 创建MIPLDataset对象，并使用DataLoader生成训练数据加载器，这里根据x_tr是否已经是张量来决定是否使用自定义的mil_collate_fn16。
            # todo 暂时不打乱顺序
            mipl_dataset_tr = MIPLDataset(x_tr, s_bag_tr, y_bag_tr, cov_b) 
            train_loader = DataLoader(mipl_dataset_tr, batch_size=args.bs, shuffle=False, 
                                    collate_fn=None if torch.is_tensor(x_tr) else mil_collate_fn, 
                                    generator=g)
            num_bags_tr = mipl_dataset_tr.__len__()

            # Model Training
            # 模型的train方法训练入口。
            model.train(train_loader, num_bags_tr, weight_list=alpha_list)

            # Model Prediction
            # 在测试集上进行预测，并计算准确率。
            if args.cuda:
                y_pred = model.predict(x_te).cpu().numpy()
            else:
                y_pred = model.predict(x_te).numpy()

            # Evaluate Performance
            # 记录当前折叠的准确率，并释放内存以准备下一个折叠。
            acc_te = accuracy_score(y_bag_te, y_pred)
            logging.info('test_acc: {:.3f}'.format(acc_te))
            print('test_acc: {:.3f}'.format(acc_te))
            accuracy[trial_i, num_i] = acc_te
            num_i += 1

            # Clear Memory
            torch.cuda.empty_cache()
            model = None
        # 7. 结果汇总
        # 在每个试验结束后，计算并记录该试验下所有折叠的平均准确率和标准差。
        logging.info("The mean and std of accuracy at {}-th times {} folds: {}, {}".format(
            trial_i+1, num_i, np.around(np.mean(accuracy[trial_i, :]), 3), 
            np.around(np.std(accuracy[trial_i, :]), 3)))
        print("The mean and std of accuracy at {}-th times {} folds: {}, {}".format(
            trial_i+1, num_i, np.around(np.mean(accuracy[trial_i, :]), 3), 
            np.around(np.std(accuracy[trial_i, :]), 3)))

    # 实验结束后，计算所有试验的总体平均准确率和标准差，并记录总运行时间。
    # 最后，输出实验结束的信息。
    logging.info("\t================================ END ========================================")
    print("End Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    end_time = time.time()
    logging.info('The mean and std of accuracy at {} times {} folds: {}, {}'.format(
            num_trial, num_i, np.around(np.mean(accuracy), 3), np.around(np.std(accuracy), 3)))
    print('The mean and std of accuracy at {} times {} folds: {}, {}'.format(
            num_trial, num_i, np.around(np.mean(accuracy), 3), np.around(np.std(accuracy), 3)))
    logging.info('\tRunning time is {} seconds.'.format(end_time - start_time))
    print('\tRunning time is {} seconds.'.format(end_time - start_time))
    logging.info('Training is finished.') 
    print('Training is finished.')