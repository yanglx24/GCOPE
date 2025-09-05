from fastargs.decorators import param
import torch
from torch_geometric.data import Batch
from copy import deepcopy
from data.utils import load_datasets
from model.graph_coordinator import GraphCoordinator
from torch_cluster import random_walk


@param("general.cache_dir")
@param("pretrain.cross_link")
@param("pretrain.cl_init_method")
@param("pretrain.cross_link_ablation")
@param("pretrain.dynamic_edge")
@param("pretrain.dynamic_prune")
@param("pretrain.split_method")
def get_clustered_data(
    datasets,
    cache_dir,
    cross_link,
    cl_init_method="learnable",
    cross_link_ablation=False,
    dynamic_edge="none",
    dynamic_prune=0.0,
    split_method="RandomWalk",
):
    data_list = [data for data in load_datasets(datasets)]
    import pdb

    pdb.set_trace()
    data = Batch.from_data_list(data_list)

    print(
        f"Isolated graphs have total {data.num_nodes} nodes, each dataset added {cross_link} graph coordinators"
    )
    gco_model = None

    if cross_link > 0:
        num_graphs = data.num_graphs
        graph_node_indices = []

        for graph_index in range(num_graphs):
            node_indices = (data.batch == graph_index).nonzero(as_tuple=False).view(-1)
            graph_node_indices.append(node_indices)

        new_index_list = [i for i in range(num_graphs)] * cross_link

        if cl_init_method == "learnable":

            gco_model = GraphCoordinator(data.num_node_features, len(new_index_list))
            data.x = gco_model.add_learnable_features_with_no_grad(data.x)

        data.batch = torch.cat(
            [data.batch, torch.tensor([new_index_list]).squeeze(0)], dim=0
        )

        if cross_link == 1:
            for node_graph_index in new_index_list:
                node_indices_corresponding_graph = (
                    (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                )
                new_node_index = node_indices_corresponding_graph[-1]

                new_edges = torch.cartesian_prod(
                    torch.tensor([new_node_index]),
                    node_indices_corresponding_graph[:-1],
                )
                data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)

                new_edges = torch.cartesian_prod(
                    node_indices_corresponding_graph[:-1],
                    torch.tensor([new_node_index]),
                )
                data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)

        if cross_link_ablation == False:
            all_added_node_index = [
                i for i in range(data.num_nodes - len(new_index_list), data.num_nodes)
            ]
            for list_index, new_node_index in enumerate(all_added_node_index[:-1]):
                other_added_node_index_list = [
                    index for index in all_added_node_index if index != new_node_index
                ]
                new_edges = torch.cartesian_prod(
                    torch.tensor([new_node_index]),
                    torch.tensor(other_added_node_index_list),
                )
                data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
    print(
        f"Unified graph has {data.num_nodes} nodes, each graph includes {cross_link} graph coordinators"
    )

    raw_data = deepcopy(data)

    if split_method == "RandomWalk":
        split_ratio = 0.1
        walk_length = 30
        all_random_node_list = torch.randperm(data.num_nodes)
        selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)
        random_node_list = all_random_node_list[:selected_node_num_for_random_walk]
        walk_list = random_walk(
            data.edge_index[0],
            data.edge_index[1],
            random_node_list,
            walk_length=walk_length,
        )

        graph_list = []
        skip_num = 0
        for walk in walk_list:
            subgraph_nodes = torch.unique(walk)
            if len(subgraph_nodes) < 5:
                skip_num += 1
                continue
            subgraph_data = data.subgraph(subgraph_nodes)

            graph_list.append(subgraph_data)

        print(
            f"Total {len(graph_list)} subgraphs with nodes more than 5, and there are {skip_num} skipped subgraphs with nodes less than 5."
        )
    return graph_list, gco_model


def update_graph_list_param(graph_list, gco_model):

    count = 0
    for graph_index, graph in enumerate(graph_list):
        for index, param_value in enumerate(gco_model.last_updated_param):
            match_info = torch.where(graph.x == param_value)
            if match_info[0].shape[0] != 0:
                target_node_indice = match_info[0].unique()[-1].item()
                graph.x[target_node_indice] = gco_model.learnable_param[index].data
                count += 1
    updated_graph_list = graph_list
    return updated_graph_list
