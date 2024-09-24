import os
import time
import math
import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import sklearn.metrics as sk_metrics

from IPython import embed
from tqdm import trange, tqdm
from layers import StackedGCN
from torch.autograd import Variable
from sklearn.metrics import f1_score
from torch_geometric.nn import Linear
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.nn import GCNConv, SAGEConv


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)

        self.lin = Linear(32, out_dim)

    def forward(self, x, edge_index):
        '''
        TODO: define forward process of GCN
        Args:
            x: tensor: (node_number, in_channels)
            edge_index: adjacent matrix with (2, node_number)
        Returns: out: (node_number, out_channels)
        '''
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        out = self.lin(x)
        return torch.log_softmax(out, dim=-1)


class GraphSAGEBasic(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)

        x = self.conv2(x, adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)

        x = self.conv3(x, adj_t)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        return torch.log_softmax(x, dim=-1)


class ClusterGCNTrainer(object):
    """
    Training a ClusterGCN.
    """
    def __init__(self, args, clustering_machine):
        """
        :param ags: Arguments object.
        :param clustering_machine:
        """  
        self.args = args
        self.clustering_machine = clustering_machine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_model()
        self.save_root = '/public_bme/home/liujm/Project/IC_Project'

    def create_model(self):
        """
        Creating a StackedGCN and transferring to CPU/GPU.
        """
        # self.model = StackedGCN(self.args, self.clustering_machine.feature_count, self.clustering_machine.class_count)
        #
        # self.model = self.model.to(self.device)
        # TODO: define GCN model
        self.model = GCN(self.clustering_machine.feature_count,32, self.clustering_machine.class_count)
        self.test_model = GCN(self.clustering_machine.feature_count, 32, self.clustering_machine.class_count)

        # # TODO: define GraphSAGE model using PyG package
        # self.model = GraphSAGE(in_channels=self.clustering_machine.feature_count, hidden_channels=16, num_layers=2, out_channels=self.clustering_machine.class_count, dropout=0.2)
        # self.test_model = GraphSAGE(in_channels=self.clustering_machine.feature_count, hidden_channels=32, num_layers=3, out_channels=self.clustering_machine.class_count, dropout=0.2)

        # TODO: define GraphSAGE using Full-batch PyG model
        # self.model = GraphSAGEBasic(self.clustering_machine.feature_count, 16, self.clustering_machine.class_count, dropout=0.2)
        self.model = self.model.to(self.device)

    def do_forward_pass(self, cluster):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.clustering_machine.sg_train_nodes[cluster].to(self.device)
        train_nodes = train_nodes==1
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()

        predictions = self.model(features, edges)
        average_loss = torch.nn.functional.nll_loss(predictions[train_nodes], target[train_nodes])
        # predictions = self.model(features, edges, batch_size=1, num_sampled_nodes_per_hop=2)
        # average_loss = torch.nn.functional.cross_entropy(predictions[train_nodes], target[train_nodes])

        node_count = train_nodes.shape[0]
        return average_loss, node_count


    def _feature_embedding_update(self, cluster):
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.clustering_machine.sg_train_nodes[cluster].to(self.device)
        train_nodes = train_nodes==1
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()

        predictions = self.model(features, edges)
        return predictions.cpu().detach().numpy()

    def update_average_loss(self, batch_average_loss, node_count):

        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster. 
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item()*node_count
        self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss/self.node_count_seen
        return average_loss

    def do_test_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        test_nodes = self.clustering_machine.sg_test_nodes[cluster].to(self.device)
        test_nodes = test_nodes==1
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        target = target[test_nodes]
        prediction = self.test_model(features, edges)
        prediction = prediction[test_nodes,:]

        return prediction, target

    def do_val_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        val_nodes = self.clustering_machine.sg_val_nodes[cluster].to(self.device)
        val_nodes = val_nodes==1
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        target = target[val_nodes]
        prediction = self.model(features, edges)
        prediction = prediction[val_nodes,:]

        return prediction, target

    def train(self, train_idx=0):
        """
        Training a model.
        """
        if not os.path.exists(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_FULL')):
            os.mkdir(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_FULL'))
        if not os.path.exists(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_FULL', str(train_idx))):
            os.mkdir(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_FULL', str(train_idx)))
        rec = list()

        print("Training started.\n")
        epochs = trange(self.args.epochs, desc = "Train Loss")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        best_f1 = 0
        loss_tmp = 10e10

        tic = time.time()
        for epoch in epochs:
            random.shuffle(self.clustering_machine.clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            # for cluster in self.clustering_machine.clusters:
            for cluster in self.clustering_machine.clusters:
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_forward_pass(cluster)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)

            tmp_f1 = self.validation()
            if best_f1 < tmp_f1:
                best_f1 = tmp_f1
                self.save_path = os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_FULL', str(train_idx), str(epoch)+'.pth.gz')
                torch.save(self.model.state_dict(), self.save_path)

            rec.append([epoch, average_loss, tmp_f1])
            epochs.set_description("Train Loss: %g" % round(average_loss,4))
        toc = time.time()

        df = pd.DataFrame(rec, columns=['epoch', 'training_loss', 'validation_f1'])
        df.to_csv(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_FULL', str(train_idx), str(epoch)+'.csv'), index=False)
        return toc-tic


    def train_random(self, train_idx=0, ratio=0.2):
        """
        Training a model.
        """
        print("Training started.\n")
        if not os.path.exists(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_RS')):
            os.mkdir(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_RS'))
        if not os.path.exists(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_RS', str(train_idx))):
            os.mkdir(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_RS', str(train_idx)))

        rec = list()
        epochs = trange(self.args.epochs, desc = "Train Loss")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        best_f1 = 0
        loss_tmp = 10e10
        clusters_tmp = random.sample(self.clustering_machine.clusters, math.floor(self.args.cluster_number*ratio))
        # print('Number of full clusters: {}, Number of training clusters: {}'.format(self.clustering_machine.clusters, clusters_tmp))

        tic = time.time()
        for epoch in epochs:
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for idx in clusters_tmp:
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_forward_pass(idx)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)

            tmp_f1 = self.validation()
            if best_f1 < tmp_f1:
                best_f1 = tmp_f1
                self.save_path = os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_RS', str(train_idx), str(epoch)+'.pth.gz')
                torch.save(self.model.state_dict(), self.save_path)

            epochs.set_description("Train Loss: %g" % round(average_loss, 4))
            rec.append([epoch, average_loss, tmp_f1])

        toc = time.time()
        df = pd.DataFrame(rec, columns=['epoch', 'training_loss', 'validation_f1'])
        df.to_csv(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_RS', str(train_idx), str(epoch)+'.csv'), index=False)
        return toc-tic


    def train_random_diversity(self, train_idx=0):
        """
        Training a model.
        """
        print("Training started.\n")
        if not os.path.exists(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_DS')):
            os.mkdir(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_DS'))
        if not os.path.exists(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_DS', str(train_idx))):
            os.mkdir(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_DS', str(train_idx)))

        rec = list()
        epochs = trange(self.args.epochs, desc = "Train Loss")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        best_f1 = 0
        loss_tmp = 10e10
        clusters = self.clustering_machine.clusters
        cluster = random.choice(clusters)
        diversity_tmp = self.clustering_machine._diversity_matrix(cluster)

        tic = time.time()
        for epoch in epochs:
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for idx in diversity_tmp:
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_forward_pass(idx)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)

            tmp_f1 = self.validation()
            if best_f1 < tmp_f1:
                best_f1 = tmp_f1
                self.save_path = os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_DS', str(train_idx), str(epoch)+'.pth.gz')
                torch.save(self.model.state_dict(), self.save_path)
            epochs.set_description("Train Loss: %g" % round(average_loss, 4))
            rec.append([epoch, average_loss, tmp_f1])

        toc = time.time()
        df = pd.DataFrame(rec, columns=['epoch', 'training_loss', 'validation_f1'])
        df.to_csv(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_DS', str(train_idx), str(epoch)+'.csv'), index=False)

        return toc-tic


    def train_random_diversity_learnable(self, train_idx=0):
        """
        Training a model.
        """
        print("Training started.\n")
        if not os.path.exists(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_DDS')):
            os.mkdir(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_DDS'))

        if not os.path.exists(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_DDS', str(train_idx))):
            os.mkdir(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_DDS', str(train_idx)))

        rec = list()
        epochs = trange(self.args.epochs, desc = "Train Loss")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        best_f1 = 0
        loss_tmp = 10e10
        clusters = self.clustering_machine.clusters
        cluster = random.choice(clusters)
        diversity_tmp = self.clustering_machine._diversity_matrix(cluster)

        tic = time.time()
        for epoch in epochs:
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for idx in diversity_tmp:
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_forward_pass(idx)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)

            tmp_f1 = self.validation()
            if best_f1 < tmp_f1:
                best_f1 = tmp_f1
                self.save_path = os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_DDS', str(train_idx),str(epoch)+'.pth.gz')
                torch.save(self.model.state_dict(), self.save_path)

            # if loss_tmp > average_loss:
            #     loss_tmp = average_loss
            #     self.save_path = os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name, str(train_idx), str(epoch)+'.pth.gz')
            #     torch.save(self.model.state_dict(), self.save_path)

            if epoch == 300 or epoch == 600:
                updated_embedding = {}
                for idx in self.clustering_machine.clusters:
                    updated_embedding[idx] = self._feature_embedding_update(idx)
                diversity_tmp = self.clustering_machine._diversity_matrix_update(cluster, updated_embedding)

            epochs.set_description("Train Loss: %g" % round(average_loss, 4))
            rec.append([epoch, average_loss, tmp_f1])

        toc = time.time()
        df = pd.DataFrame(rec, columns=['epoch', 'training_loss', 'validation_f1'])
        df.to_csv(os.path.join(self.args.save_root, self.args.save_folder, self.args.save_name+'_DDS', str(train_idx), str(epoch)+'.csv'), index=False)

        return toc-tic

    def validation(self):
        self.model.eval()
        self.predictions = []
        self.targets = []
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_val_prediction(cluster)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        self.targets = np.concatenate(self.targets)
        self.predictions = np.concatenate(self.predictions).argmax(1)
        score = f1_score(self.targets, self.predictions, average="macro")
        self.model.train()
        return score

    def test(self):
        """
        Scoring the test and printing the F-1 score.
        """
        self.test_model = self.model.to(self.device)
        # self.test_model.load_state_dict(torch.load(self.save_path))
        self.test_model.eval()

        self.predictions = []
        self.targets = []
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_test_prediction(cluster)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        self.targets = np.concatenate(self.targets)
        self.predictions = np.concatenate(self.predictions).argmax(1)

        precision_score = sk_metrics.precision_score(self.targets, self.predictions, average="macro")
        recall_score = sk_metrics.recall_score(self.targets, self.predictions, average="macro")
        f1_score = sk_metrics.f1_score(self.targets, self.predictions, average="macro")

        print("\nPrecision score: {:.4f}".format(precision_score))
        print("\nRecall score: {:.4f}".format(recall_score))
        print("\nF-1 score: {:.4f}".format(f1_score))
        return precision_score, recall_score, f1_score
