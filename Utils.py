import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,cohen_kappa_score
import os
import shutil
import logging
import datetime


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def acc_target(predicted_labels, true_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    return accuracy, precision, recall, f1, kappa




class exp_master():
    def __init__(self, base_folder, arg):
        self.base_folder = base_folder
        self.arg=arg
        self.model_name = arg.model_name
        self.dest_path = self._create_exp_folder()


    def _create_exp_folder(self):
        # 检查exp文件夹是否存在，如果不存在则创建
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)

        # 遍历已存在的文件夹，获取最大的序号
        existing_folders = [name for name in os.listdir(self.base_folder) if name.startswith('exp')]
        if existing_folders:
            max_index = max(int(name[3:]) for name in existing_folders)
        else:
            max_index = 0

        # 创建新的文件夹
        new_folder_name = f'exp{max_index + 1}'
        new_folder_path = os.path.join(self.base_folder, new_folder_name)
        os.makedirs(new_folder_path)

        return new_folder_path


    def folder_copy(self):
        shutil.copy(self.model_name+'.py', os.path.join(self.dest_path, self.model_name+'.py'))
        shutil.copy('main.py', os.path.join(self.dest_path, 'main.py'))
        shutil.copy('Train.py', os.path.join(self.dest_path, 'Train.py'))


    def pre_train(self, train_log_name='training.log'):
        self.folder_copy()
        log_file_name = os.path.join(self.dest_path, train_log_name)
        logging.basicConfig(filename=log_file_name, level=logging.INFO, format='%(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # 设置控制台输出的日志级别
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger('').addHandler(console_handler)
        current_time = datetime.datetime.now()
        logging.info(f'Training start at : {current_time}')


    def get_dest_path(self):
        return self.dest_path


def l1_regularization(params, lambda_l1):
    l1_norm = torch.sum(torch.abs(params))
    return lambda_l1 * l1_norm


def Adj_divide(input_A, num_node, time_length, lambda_l1, enhance_hyper):
    l1 = 0
    for row in range(time_length):
        for col in range(time_length):
            row_start = row * num_node
            row_end = (row + 1) * num_node
            col_start = col * num_node
            col_end = (col + 1) * num_node
            temp = input_A[row_start:row_end, col_start:col_end]
            temp_l1 = l1_regularization(temp, lambda_l1) * (enhance_hyper ** (abs(row - col)))
            l1 = l1 + temp_l1
    return l1





if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    A = Mask_Matrix(device, 19, 3, 0.8)
    l1 = Adj_divide(A, 19, 3, 1e-4,2)
    print(l1)
