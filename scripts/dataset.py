from tqdm.auto import tqdm

import torch
import torch_geometric

from torch.nn import Linear
from torch.nn import functional as F 
from torch.optim.lr_scheduler import ExponentialLR

from torch_geometric.utils.smiles import from_smiles
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm

from scripts.reader import CUT_OFF_THRESHOLD


def prepare_graph_dataset(
    smile_list: list[str], 
    pIC50_list: list[float], 
    problem_type: str = "classification",
) -> list[torch_geometric.data]:
    """Create the 'dataset' by converting the SMILES strings to featurised 
        graphs via deepchem package, and attaching the 'y' labels to 
        the 'x' features (both as float tensors)."""
    assert len(smile_list) == len(pIC50_list)
    desc = "Converting SMILES strings to torch_geometric data: "

    # Featurise the graph data:
    data_list = []
    
    for i in tqdm(range(len(smile_list)), desc=desc):
        smiles = smile_list[i]
        graph_dataset = from_smiles(smiles)
        graph_dataset.x = graph_dataset.x.to(torch.float32)
            
        if problem_type == "regression":
            graph_dataset.y = torch.Tensor([pIC50_list[i]])

        elif problem_type == "classification":
            class_label = torch.Tensor([pIC50_list[i] > CUT_OFF_THRESHOLD])
            graph_dataset.y = class_label.long()

        else:
            raise NotImplementedError("Choose an implemented approach.")

        data_list.append(graph_dataset)
    
    return data_list


def load_into_dataloader(
    dataset: list[torch_geometric.data], 
    batch_size: int,
    shuffle: bool
) -> torch_geometric.loader.DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )
