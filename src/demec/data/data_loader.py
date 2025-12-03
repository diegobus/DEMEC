import pandas as pd
import os
import torch
from torch.utils.data import Dataset, random_split
import networkx as nx
from torch_geometric.data import Data
import pickle
from torch_geometric.utils import from_networkx
import numpy as np


class GraphStructureDataset(Dataset):

    def __init__(self, graph_dir, cid_se_csv=None, task_config=None, node_dim=1, feature_key=None):

        super().__init__()
        self.graph_dir = graph_dir
        self.feature_key = feature_key
        
        # task_config should be {task_name: csv_path}
        self.task_configs = {}
        if cid_se_csv:
            self.task_configs['side_effects'] = cid_se_csv
        if task_config:
            self.task_configs.update(task_config)
            
        self.task_cid_maps = {}
        self.task_dims = {}
        
        for task_name, csv_path in self.task_configs.items():
            df = pd.read_csv(csv_path).set_index("cid")
            # Store dimensions for model initialization
            self.task_dims[task_name] = len(df.columns)
            
            # Create mapping
            cid_map = {
                int(cid): torch.tensor(row.values, dtype=torch.float32)
                for cid, row in df.iterrows()
            }
            self.task_cid_maps[task_name] = cid_map
            
            # Keep backward compatibility for se_cols if it's the side_effects task
            if task_name == 'side_effects':
                self.se_cols = list(df.columns)

        # Load graph files
        files = os.listdir(graph_dir)
        items = []
        for file in files:
            cid = int(file.split(".")[0])
            full_file = graph_dir + file
            items.append((cid, full_file))

        self.items = sorted(items, key=lambda t: t[0])

        self.node_dim = node_dim

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        cid, full_file = self.items[idx]
        with open(full_file, "rb") as f:
            G = pickle.load(f)
        data = from_networkx(G)
        
        if self.feature_key and self.feature_key in G.nodes[list(G.nodes)[0]]:
            # Extract features from NetworkX graph using the specified key
            # We iterate over nodes to ensure order matches G.nodes() which from_networkx preserves
            features = [G.nodes[n][self.feature_key] for n in G.nodes()]
            data.x = torch.tensor(np.array(features), dtype=torch.float32)
            # Update node_dim based on actual feature size if not manually set (optional, but safer to trust init)
        else:
            data.x = torch.ones((data.num_nodes, self.node_dim), dtype=torch.float32)
            
        data.cid = torch.tensor([cid], dtype=torch.int64)
        
        # Attach targets for each task
        for task_name, cid_map in self.task_cid_maps.items():
            if cid in cid_map:
                # Attach as y_{task_name}
                # We also keep data.y for backward compatibility if it's side_effects
                target = cid_map[cid].unsqueeze(0)
                setattr(data, f"y_{task_name}", target)
                if task_name == 'side_effects':
                    data.y = target
                    
        return data


def make_splits(dataset, train=0.8, val=0.1, seed=42):
    n = len(dataset)
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=g)
