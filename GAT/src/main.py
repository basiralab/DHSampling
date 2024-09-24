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
    rec = list()

    for idx in range(15):
        clustering_machine = ClusteringMachine(args, graph, features, target)
        clustering_machine.decompose()
        gcn_trainer = ClusterGCNTrainer(args, clustering_machine)

        if args.type=='full':
            print('Random clustering training')
            tt = gcn_trainer.train(idx)
            print('Training time usage:{}'.format(tt))
        elif args.type == 'random_sampling':
            print('Random clustering training with random sampling')
            tt = gcn_trainer.train_random(idx)
            print('Training time usage:{}'.format(tt))
        elif args.type == 'diversity_sampling':
            print('Random clustering training with diversity-guided sampling')
            tt = gcn_trainer.train_random_diversity(idx)
            print('Training time usage:{}'.format(tt))
        elif args.type == 'diversity_learnable_sampling':
            print('Random clustering training with learnable diversity sampling')
            tt = gcn_trainer.train_random_diversity_learnable(idx)
            print('Training time usage:{}'.format(tt))

        precision, recall, f1 = gcn_trainer.test()

        pynvml.nvmlInit()
        # 这里的0是GPU id
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

        rec.append([precision, recall, f1, tt, meminfo.used])

    df = pd.DataFrame(rec, columns=['Precision', 'Recall', 'F1Score', 'Time', 'GPU'])
    df.to_csv(os.path.join(args.save_root, args.save_folder, args.save_name+args.type+'.csv'), index=False)


if __name__ == "__main__":
    main()
