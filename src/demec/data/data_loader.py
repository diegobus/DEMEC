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

    def __init__(self, graph_dir, cid_se_csv=None, task_config=None, node_dim=1, feature_key=None, max_side_effects=None):

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
        self.has_other_ses_map = {}  # Track drugs with non-top-N side effects

        for task_name, csv_path in self.task_configs.items():
            df = pd.read_csv(csv_path).set_index("cid")

            # For side_effects, optionally filter to top N most common
            if task_name == 'side_effects' and max_side_effects is not None:
                # Select top N side effects by prevalence (sum of occurrences)
                col_sums = df.sum(axis=0).sort_values(ascending=False)
                top_cols = col_sums.head(max_side_effects).index.tolist()
                
                # Before filtering, identify drugs with side effects NOT in top N
                for cid, row in df.iterrows():
                    total_ses = row.sum()  # Total number of side effects
                    top_n_ses = row[top_cols].sum()  # Number of top-N side effects
                    has_other = (total_ses > top_n_ses)  # Has SEs not in top N
                    self.has_other_ses_map[int(cid)] = float(has_other)
                
                # Now filter to top N
                df = df[top_cols]
                print(f"Limited side_effects to top {max_side_effects} (from {len(col_sums)} total)")
                print(f"  {sum(self.has_other_ses_map.values())} drugs have side effects not in top {max_side_effects}")

            # Store dimensions for model initialization
            # Add +1 for "has_other_SEs" flag if we filtered side effects
            if task_name == 'side_effects' and max_side_effects is not None:
                self.task_dims[task_name] = len(df.columns) + 1  # +1 for has_other_SEs
            else:
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

        # Standardize attributes to ensure consistency for PyG batching
        if len(G.nodes) > 0:
            if self.feature_key:
                for _, data_dict in G.nodes(data=True):
                    keys_to_remove = [k for k in data_dict if k != self.feature_key]
                    for k in keys_to_remove:
                        del data_dict[k]
            else:
                for _, data_dict in G.nodes(data=True):
                    data_dict.clear()

        # Remove edge attributes as they are unused and can cause batching errors
        if len(G.edges) > 0:
            for u, v, d in G.edges(data=True):
                d.clear()

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
                target = cid_map[cid].unsqueeze(0)
                
                # For side_effects with filtering, append the "has_other_SEs" flag
                if task_name == 'side_effects' and hasattr(self, 'has_other_ses_map') and len(self.has_other_ses_map) > 0:
                    has_other = torch.tensor([[self.has_other_ses_map.get(cid, 0.0)]], dtype=torch.float32)
                    target = torch.cat([target, has_other], dim=1)
                
                setattr(data, f"y_{task_name}", target)
                setattr(data, f"mask_{task_name}", torch.tensor([True], dtype=torch.bool))
                
                # Backward compatibility
                if task_name == 'side_effects':
                    data.y = target
            else:
                # Missing label: fill with zeros and mask out
                dim = self.task_dims[task_name]
                dummy = torch.zeros((1, dim), dtype=torch.float32)
                setattr(data, f"y_{task_name}", dummy)
                setattr(data, f"mask_{task_name}", torch.tensor([False], dtype=torch.bool))
                
                if task_name == 'side_effects':
                    data.y = dummy
                    
        return data


def make_splits(dataset, train=0.8, val=0.1, seed=42):
    n = len(dataset)
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=g)
