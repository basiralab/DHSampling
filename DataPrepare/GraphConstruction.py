import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython import embed
from sklearn import metrics as sk


def _sample_norm(data):
    '''
    TODO: functions to perform z-score normalization on matrix NxM
    Args:
        data: numpy array with (NxM)
    Returns: normalized data with (NxM) by z-score normalization
    '''
    data_norm = (data.T-data.mean(1))/data.std(1)
    return data_norm.T


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting for convert MedMNIST data to graph')
    parser.add_argument('--medmnist_npy', type=str, default='/path/2/medmnist/npy', help='specify original medmnist data file')
    parser.add_argument('--medmnist_graph', type=str, default='/path/2/medmnist/graph', help='specify training file list')

    args = parser.parse_args()

    data_flags_2d = ['organcmnist', 'organsmnist']

    for data_flag in tqdm(data_flags_2d):
        print('Start process data: {}'.format(data_flag))
        num_features = 28 * 28
        target_folder = os.path.join(args.medmnist_graph, data_flag)
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

        # TODO: load training data and corresponding label
        train_images = np.load("{}/{}".format(args.medmnist_npy, data_flag) + "/train_images.npy")
        train_data = train_images.reshape(train_images.shape[0],num_features).astype('float')
        train_labels = np.load("{}/{}".format(args.medmnist_npy, data_flag) + "/train_labels.npy")
        num_train = train_images.shape[0]
        train_flag = np.ones((num_train)) * 1

        # TODO: load validation data and corresponding label
        val_images = np.load("{}/{}".format(args.medmnist_npy, data_flag) + "/val_images.npy")
        val_data = val_images.reshape(val_images.shape[0],num_features).astype('float')
        val_labels = np.load("{}/{}".format(args.medmnist_npy, data_flag) + "/val_labels.npy")
        num_val = val_images.shape[0]
        validation_flag = np.ones((num_val)) * 2

        # TODO: load test data and corresponding label
        test_images = np.load("{}/{}".format(args.medmnist_npy, data_flag) + "/test_images.npy")
        test_data = test_images.reshape(test_images.shape[0],num_features).astype('float')
        test_labels = np.load("{}/{}".format(args.medmnist_npy, data_flag) + "/test_labels.npy")
        num_test = test_images.shape[0]
        test_flag = np.ones((num_test)) * 3

        # TODO: Concatenate data of training, validation, and testing together
        num_data = num_train + num_val + num_test
        data_feat = np.concatenate((train_data, val_data, test_data), axis = 0)
        data_feat = _sample_norm(data_feat)

        data_label = np.concatenate((train_labels, val_labels, test_labels), axis = 0)
        data_label = data_label.reshape((data_label.shape[0], -1))

        data_flag = np.concatenate((train_flag, validation_flag, test_flag), axis = 0)
        data_flag = data_flag.reshape((data_label.shape[0], -1))

        # TODO: Construct Adjacency Matrix
        A = sk.pairwise.cosine_similarity(data_feat, data_feat)
        A = np.abs(A)
        edge_threshold = np.percentile(A, 99.6)
        A = A >= edge_threshold

        edge_index = list()

        for i in tqdm(range(num_data)):
            for j in range(num_data):
                if (i != j and A[i][j] == True):
                    edge_index.append([i, j])

        df_edge = pd.DataFrame(edge_index, columns=['id1', 'id2'])
        df_edge.to_csv('{}/{}'.format(target_folder, 'edges.csv'), index=False)

        # TODO: convert data to target csv file
        target_rec = list()
        for idx in tqdm(range(num_data)):
            target_rec.append([idx, data_label[idx]])
        df_target = pd.DataFrame(target_rec, columns=['node_id', 'target'])
        df_target.to_csv('{}/{}'.format(target_folder, 'target.csv'), index=False)

        # TODO: convert node feature to csv file
        np.save('{}/{}'.format(target_folder, 'features.npy'), data_feat)

        # TODO: convert data to target csv file
        split_rec = list()
        for idx in tqdm(range(num_data)):
            split_rec.append([idx, data_flag[idx]])
        df_split = pd.DataFrame(split_rec, columns=['node_id', 'flag'])
        df_target.to_csv('{}/{}'.format(target_folder, 'flag.csv'), index=False)