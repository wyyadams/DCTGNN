'''
main for PFCSTGNN
'''


import datetime
from Utils import exp_master
import argparse
from Train import train
import logging


def main(args):
    exp_log = exp_master('exp', args)
    exp_log.pre_train()
    dest = exp_log.get_dest_path()
    logging.info("DCTGNN Argument values:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    train(args)
    current_time = datetime.datetime.now()
    logging.info(f'Training end at : {current_time} \n')
    logging.info(f'Info saved in : {dest} \n\n\n')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PFCSTGNN')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--dataset_path', type=str, default='MODMA_Dataset\MODMA_demo_dataset.npz', help='')
    parser.add_argument('--std_flag', type=bool, default=False,help='HUSM:True, MODMA:False')
    parser.add_argument('--model_name', type=str, default='DCTGNN', help='')
    parser.add_argument('--win_len', type=int, default=250, help='')

    # Model parameter
    parser.add_argument('--chan_num', type=int, default=19, help='')
    parser.add_argument('--tlen', type=int, default=4, help='')
    parser.add_argument('--fea_emb', type=int, default=128, help='Temperature of QK')
    parser.add_argument('--GNN_out_dim', type=int, default=32, help='')
    parser.add_argument('--atten_hid_r', type=int, default=4, help='')
    parser.add_argument('--time_win1', type=int, default=2, help='')
    parser.add_argument('--time_win2', type=int, default=3, help='')
    parser.add_argument('--n_reg',type=int, default=7,help='')
    parser.add_argument('--s1', type=int, default=1, help='')
    parser.add_argument('--s2', type=int, default=1, help='')
    parser.add_argument('--K_GCN', type=int, default=2)
    parser.add_argument('--K_R_GCN', type=int, default=2)
    parser.add_argument('--P_S', type=int, default=3, help='Partition Scheme in R2R 0=General,1=Hemispehre,2=Frontal')
    parser.add_argument('--dn', type=str, default='MODMA',help='MODMA or HUSM')
    parser.add_argument('--T_flag', type=bool, default=True)
    parser.add_argument('--S_flag', type=bool, default=False)
    parser.add_argument('--init_st', type=str, default='xavier')
    parser.add_argument('--init_qk', type=str, default='kaiming')
    parser.add_argument('--atten_opt', type=bool, default=True)

    # Training parameter
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-2, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--total_epoch', type=int, default=100, help='')
    parser.add_argument('--l1_lambda', type=float, default=1e-6, help='')
    parser.add_argument('--enhance_hyper', type=float, default=2, help='')



    args = parser.parse_args()
    main(args)
