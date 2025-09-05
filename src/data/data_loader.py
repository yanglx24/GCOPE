import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    TUDataset,
    Reddit,
    AttributedGraphDataset,
    Planetoid,
    Amazon,
    FacebookPagePage,
    WordNet18RR,
    TUDataset,
    MoleculeNet,
    WebKB,
)
from data.data_custom import FB15k_237
from ogb.nodeproppred import PygNodePropPredDataset
from data.data_transform import UnifyFeatureDims, FewShotLinkSplit, Node2VecEmbedding
from data.data_process import graph_few_shot_splits, link_k_shot_split
from torch_geometric.data import Dataset
from torch_geometric.utils import to_undirected


def load_pretrain_single_graph_data(root, data_name):
    if data_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(
            root=root, name=data_name, transform=T.Compose([T.ToUndirected()])
        )
        data = dataset[0]
    elif data_name == "Computers":
        dataset = Amazon(root=root, name=data_name)
        data = dataset[0]
    elif data_name == "Reddit":
        dataset = Reddit(root=f"{root}/{data_name}")
        data = dataset[0]
    elif data_name == "FB15k_237":
        transform = Node2VecEmbedding(
            embed_dim=100,
            batch_size=128,
            walk_length=20,
            context_size=10,
            lr=0.01,
            walks_per_node=10,
            p=1.0,
            q=1.0,
            num_epochs=100,
        )
        dataset = FB15k_237(
            root=f"{root}/{data_name}", split="train", pre_transform=transform
        )
        data = dataset[0]
    elif data_name == "PPI":
        dataset = AttributedGraphDataset(root=root, name=data_name.lower())
        data = dataset[0]
    elif data_name in ["Wisconsin", "Cornell", "Texas"]:
        dataset = WebKB(root=root, name=data_name.lower())
        data = dataset[0]
    else:
        raise ValueError("Invalid data_name")
    data = UnifyFeatureDims(50)(data)
    if data.edge_weight is None:
        data.edge_weight = torch.ones_like(data.edge_index[0]).float()
    data.edge_index, data.edge_weight = to_undirected(
        data.edge_index, data.edge_weight, num_nodes=data.num_nodes
    )
    return data


def load_pretrain_multi_graph_data(root, data_name):
    if data_name == "HIV":
        dataset = MoleculeNet(root=root, name=data_name, transform=UnifyFeatureDims(50))
    elif data_name == "PROTEINS":
        dataset = TUDataset(root=root, name=data_name, transform=UnifyFeatureDims(50))
    else:
        raise ValueError("Invalid data_name")
    dataset = GraphDataset(dataset)
    return dataset


class Flatten(T.BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        data.y = data.y.reshape(-1)
        return data


def load_few_shot_single_graph_data(
    root, data_name, k_shot, num_splits, num_val=0.1, num_test=0.2
):
    transform = T.RandomNodeSplit(
        split="test_rest",
        num_splits=num_splits,
        num_train_per_class=k_shot,
        num_val=num_val,
        num_test=num_test,
    )
    if data_name == "ogbn-arxiv":
        dataset = PygNodePropPredDataset(
            root=root,
            name=data_name,
            transform=T.Compose([T.ToUndirected(), Flatten(), transform]),
        )
    elif data_name == "PubMed":
        dataset = Planetoid(root=root, name=data_name, transform=transform)
    elif data_name in ["Computers", "Photo"]:
        dataset = Amazon(root, data_name, transform=transform)
    elif data_name == "Reddit":
        dataset = Reddit(f"{root}/{data_name}", transform=transform)
    elif data_name == "FacebookPagePage":
        dataset = FacebookPagePage(f"{root}/{data_name}", transform=transform)
    elif data_name == "PPI":
        dataset = AttributedGraphDataset(root, name=data_name.lower())
    else:
        raise ValueError("Invalid data_name")
    data = dataset[0]
    if data.edge_weight is None:
        data.edge_weight = torch.ones_like(data.edge_index[0]).float()
    data = UnifyFeatureDims(50)(data)
    data.edge_index, data.edge_weight = to_undirected(
        data.edge_index, data.edge_weight, num_nodes=data.num_nodes
    )
    return dataset, data


def load_few_shot_multi_graph_data(
    root, data_name, k_shot, num_splits, num_val=0.5, num_test=0.5
):
    """Just for single class classification"""
    if data_name in ["PROTEINS", "MUTAG", "ENZYMES"]:
        dataset = TUDataset(root, data_name, transform=UnifyFeatureDims(50))
    elif data_name in ["PCBA", "HIV"]:
        dataset = MoleculeNet(root, data_name, transform=UnifyFeatureDims(50))
    else:
        raise ValueError("Invalid data_name")
    dataset = GraphDataset(dataset)
    train_mask, val_mask, test_mask = graph_few_shot_splits(
        dataset, k_shot, num_val, num_splits
    )
    return dataset, train_mask, val_mask, test_mask


def load_few_shot_link_graph_data(
    root, data_name, k_shot, num_splits, num_val=0.1, num_test=0.2
):
    if data_name == "WordNet18RR":
        transform_split = FewShotLinkSplit(k_shot, num_splits, num_val, num_test)
        transform_x = Node2VecEmbedding(
            embed_dim=100,
            batch_size=128,
            walk_length=20,
            context_size=10,
            lr=0.01,
            walks_per_node=10,
            p=1.0,
            q=1.0,
            num_epochs=100,
        )
        dataset = WordNet18RR(
            f"{root}/{data_name}",
            pre_transform=T.Compose([transform_split, transform_x]),
        )
        data = dataset[0]
    elif data_name == "FB15k_237":
        transform_split = FewShotLinkSplit(k_shot, num_splits, num_val, num_test)
        transform_x = Node2VecEmbedding(
            embed_dim=100,
            batch_size=128,
            walk_length=20,
            context_size=10,
            lr=0.01,
            walks_per_node=10,
            p=1.0,
            q=1.0,
            num_epochs=100,
        )
        dataset = FB15k_237(
            f"{root}/{data_name}",
            split="train",
            pre_transform=T.Compose([transform_split, transform_x]),
        )
        data = dataset[0]
    else:
        raise ValueError("Invalid data_name")
    data = UnifyFeatureDims(50)(data)
    if data.edge_weight is None:
        data.edge_weight = torch.ones_like(data.edge_index[0]).float()
    train_mask, val_mask, test_mask = link_k_shot_split(
        data, k_shot, num_splits, num_val, num_test
    )
    return dataset, data, (train_mask, val_mask, test_mask)


class GraphDataset(Dataset):
    def __init__(self, dataset: Dataset):
        """

        :param dataset: Graph-level dataset
        """
        super(GraphDataset, self).__init__()
        self.dataset = dataset

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx]
        data.x = data.x.float()
        data.y = data.y.long().reshape(-1)
        if data.edge_weight is None:
            data.edge_weight = torch.ones_like(data.edge_index[0]).float()
        return data
