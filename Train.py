
# encoding: utf-8
from Utils import *
import numpy as np
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.utils.data as Data

import time
import logging

from DCTGNN import DCTGNN


def train(arg):
    set_seed(arg.seed)

    time_start = time.time()
    kf = KFold(n_splits=10, shuffle=True, random_state=arg.seed)


    npzloader=np.load(arg.dataset_path)
    # data.shape:[N_sub,N_sample,N_electrode,tlen(Num of Time steps/Num of Sub-slices),timestamps]
    # label.shape:[N_sub,N_sample]
    data=npzloader['data']
    label=npzloader['label']
    N_sub, _, chan_num, tl,sp = data.shape
    sub_info = np.arange(N_sub)

    if arg.std_flag == True:
        min_vals = np.min(data, axis=-1, keepdims=True)
        max_vals = np.max(data, axis=-1, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        data = (data - min_vals) / range_vals


    batchsize = arg.bs
    learning_rate = arg.lr
    weight_decay = arg.wd
    total_epoch = arg.total_epoch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc_max_all, all_max_f1, all_max_pre, all_max_recall, all_max_kappa = [], [], [], [], []


    for fold, (train_idx, test_idx) in enumerate(kf.split(sub_info)):

        train_data = np.array([data[i] for i in train_idx]).reshape(-1, chan_num, tl, sp)
        train_label = np.array([label[i] for i in train_idx]).reshape(-1)
        test_data = np.array([data[i] for i in test_idx]).reshape(-1, chan_num, tl, sp)
        test_label = np.array([label[i] for i in test_idx]).reshape(-1)

        print(fold, train_data.shape, train_label.shape, test_data.shape, test_label.shape)

        train_data, train_label, test_data, test_label = map(torch.from_numpy,
                                                             [train_data, train_label, test_data, test_label])
        train_set = Data.TensorDataset(train_data, train_label)
        test_set = Data.TensorDataset(test_data, test_label)
        train_loader = Data.DataLoader(train_set, batch_size=batchsize, shuffle=True)
        test_loader = Data.DataLoader(test_set, batch_size=batchsize, shuffle=True)

        model = DCTGNN(device=device, tlen=arg.tlen, time_window1=arg.time_win1, time_window2=arg.time_win2,
                       feat_emb=arg.fea_emb, chan_num=arg.chan_num,
                       gnn_out_chan=arg.GNN_out_dim, atten_hid_r=arg.atten_hid_r,
                       atten_opt=arg.atten_opt,
                       T_flag=arg.T_flag, S_flag=arg.S_flag, init_method=arg.init_st, init_qk=arg.init_qk,
                       K_GCN=arg.K_GCN, K_R_GCN=arg.K_R_GCN, P_s=arg.P_S, dn=arg.dn, stride1=arg.s1,
                       stride2=arg.s2).to(device)

        print(f'model at {next(model.parameters()).device}')

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_data_size, test_data_size = len(train_set), len(test_set)
        train_loss_history, val_loss_history = [], []
        train_acc_history, val_acc_history = [], []
        val_acc_max, val_f1_max, val_pre_max, val_recall_max, val_kappa_max = 0.0, 0.0, 0.0, 0.0, 0.0
        logging.info('\n\n'+'=' * 20 + f'fold:{fold}' + '=' * 20)


        for epoch in range(total_epoch):

            model.train()

            train_epoch_loss = 0.0
            epoch_train_acc_num = 0
            for i, (features, labels) in enumerate(train_loader):
                features, labels = features.float().to(device), labels.long().to(device)


                optimizer.zero_grad()
                feat, outputs, A1, A2, RA1, RA2 = model(features)

                l1_loss_1 = Adj_divide(A1, arg.chan_num, arg.time_win1, arg.l1_lambda, arg.enhance_hyper)
                l1_loss_2 = Adj_divide(A2, arg.chan_num, arg.time_win2, arg.l1_lambda, arg.enhance_hyper)
                l1_loss_R1 = Adj_divide(RA1, arg.n_reg, arg.time_win1, arg.l1_lambda, arg.enhance_hyper)
                l1_loss_R2 = Adj_divide(RA2, arg.n_reg, arg.time_win2, arg.l1_lambda, arg.enhance_hyper)

                loss = criterion(outputs, labels) + l1_loss_1 + l1_loss_2+l1_loss_R1+l1_loss_R2
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                acc_num = (predicted == labels).sum()
                epoch_train_acc_num += acc_num

            train_epoch_loss /= len(train_loader)
            train_loss_history.append(train_epoch_loss)
            train_acc = epoch_train_acc_num / train_data_size
            train_acc_history.append(train_acc.item())

            model.eval()
            with torch.no_grad():
                True_label = []
                Predict_label = []
                for i, (features, labels) in enumerate(test_loader):
                    features = features.float().to(device)
                    _, outputs, _, _,_,_ = model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    Predict_label.extend(predicted.cpu().tolist())
                    True_label.extend(labels.tolist())
                val_acc, precision, recall, val_f1, val_kappa = acc_target(Predict_label, True_label)

                if (val_acc > val_acc_max):
                    val_acc_max = val_acc
                    val_pre_max = precision
                    val_recall_max = recall
                    val_f1_max = val_f1
                    val_kappa_max=val_kappa
                    val_acc_max_epoch = epoch + 1
                    logging.info(
                        f'Epoch {epoch + 1}, Train_loss:{train_epoch_loss:.3f}, Train_acc:{train_acc:.3f}, Val_acc:{val_acc:.3f},'
                        f' ACC:{val_acc_max:.3f} , F1:{val_f1_max:.3f} , PRE:{val_pre_max:.3f} , RECALL:{val_recall_max:.3f} , KAPPA:{val_kappa_max:.3f} in[{val_acc_max_epoch}]')

        acc_max_all.append(val_acc_max)
        all_max_pre.append(val_pre_max)
        all_max_recall.append(val_recall_max)
        all_max_f1.append(val_f1_max)
        all_max_kappa.append(val_kappa_max)


    acc_max_all_array = np.array(acc_max_all)
    pre_max_all_array = np.array(all_max_pre)
    recall_max_all_array = np.array(all_max_recall)
    f1_max_all_array = np.array(all_max_f1)
    kappa_max_all_array=np.array(all_max_kappa)



    time_end = time.time()
    time_sum = time_end - time_start

    logging.info('\n\n'+'=' * 20 + f'Conclusion' + '=' * 20)
    logging.info(acc_max_all_array)
    logging.info('\n')
    logging.info(f'ACC: {np.mean(acc_max_all_array)} , {np.std(acc_max_all_array)}')
    logging.info(f'PRE: {np.mean(pre_max_all_array)} , {np.std(pre_max_all_array)}')
    logging.info(f'REC: {np.mean(recall_max_all_array)} , {np.std(recall_max_all_array)}')
    logging.info(f'F1:  {np.mean(f1_max_all_array)} , {np.std(f1_max_all_array)}')
    logging.info(f'kappa:  {np.mean(kappa_max_all_array)} , {np.std(kappa_max_all_array)}')
    logging.info('\n')
    logging.info(f'Running Time: {time_sum}')
