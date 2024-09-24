import os
import torch
import pynvml
import pandas as pd
from parser import parameter_parser
from clustering import ClusteringMachine
from clustergcn import ClusterGCNTrainer
from utils import graph_reader, feature_reader, target_reader
from IPython import embed


def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting a ClusterGCN and scoring the model.
    """
    args = parameter_parser()
    if not os.path.exists(os.path.join(args.save_root, args.save_folder)):
        os.mkdir(os.path.join(args.save_root, args.save_folder))

    if not os.path.exists(os.path.join(args.save_root, args.save_folder, args.save_name)):
        os.mkdir(os.path.join(args.save_root, args.save_folder, args.save_name))

    torch.manual_seed(args.seed)

    graph = graph_reader(args.edge_path)
    features = feature_reader(args.features_path)
    target = target_reader(args.target_path)
    rec_full, rec_RS, rec_DS, rec_DDS = list(), list(), list(), list()

    for idx in range(20):
        clustering_machine = ClusteringMachine(args, graph, features, target)
        clustering_machine.decompose()

        gcn_trainer_full = ClusterGCNTrainer(args, clustering_machine)
        print('Random clustering training')
        tt = gcn_trainer_full.train(idx)
        precision, recall, f1 = gcn_trainer_full.test()
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        rec_full.append([precision, recall, f1, tt, meminfo.used])

        gcn_trainer_rs = ClusterGCNTrainer(args, clustering_machine)
        print('Random clustering training with random sampling')
        tt = gcn_trainer_rs.train_random(idx)
        precision, recall, f1 = gcn_trainer_rs.test()
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        rec_RS.append([precision, recall, f1, tt, meminfo.used])

        gcn_trainer_ds = ClusterGCNTrainer(args, clustering_machine)
        print('Random clustering training with diversity-guided sampling')
        tt = gcn_trainer_ds.train_random_diversity(idx)
        precision, recall, f1 = gcn_trainer_ds.test()
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        rec_DS.append([precision, recall, f1, tt, meminfo.used])

        gcn_trainer_dds = ClusterGCNTrainer(args, clustering_machine)
        print('Random clustering training with learnable diversity sampling')
        tt = gcn_trainer_dds.train_random_diversity_learnable(idx)
        precision, recall, f1 = gcn_trainer_dds.test()
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        rec_DDS.append([precision, recall, f1, tt, meminfo.used])

    df_full = pd.DataFrame(rec_full, columns=['Precision', 'Recall', 'F1Score', 'Time', 'GPU'])
    df_full.to_csv(os.path.join(args.save_root, args.save_folder, args.save_name+'_Full.csv'), index=False)

    df_RS = pd.DataFrame(rec_RS, columns=['Precision', 'Recall', 'F1Score', 'Time', 'GPU'])
    df_RS.to_csv(os.path.join(args.save_root, args.save_folder, args.save_name+'_RS.csv'), index=False)

    df_DS = pd.DataFrame(rec_DS, columns=['Precision', 'Recall', 'F1Score', 'Time', 'GPU'])
    df_DS.to_csv(os.path.join(args.save_root, args.save_folder, args.save_name+'_DS.csv'), index=False)

    df_DDS = pd.DataFrame(rec_DDS, columns=['Precision', 'Recall', 'F1Score', 'Time', 'GPU'])
    df_DDS.to_csv(os.path.join(args.save_root, args.save_folder, args.save_name+'_DDS.csv'), index=False)


if __name__ == "__main__":
    main()
