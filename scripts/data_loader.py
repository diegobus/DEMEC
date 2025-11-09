import pandas as pd
import os
import torch
from torch.utils.data import Dataset, random_split
import networkx as nx
from torch_geometric.data import Data
import pickle
from torch_geometric.utils import from_networkx

class GraphStructureDataset(Dataset):

    def __init__(self, graph_dir, cid_se_csv):

        super().__init__()
        self.graph_dir = graph_dir
        cid_se_df = pd.read_csv(cid_se_csv).set_index("cid")

        self.se_cols = list(cid_se_df.columns)
        self.y_table = cid_se_df.sort_index()

        self.cid_to_y = {int(cid): torch.tensor(row.values, dtype=torch.float32) 
            for cid, row in self.y_table.iterrows()}
        
        files = os.listdir(graph_dir)
        items = []
        for file in files: 
            cid = int(file.split('.')[0])
            full_file = graph_dir + file
            items.append((cid, full_file))

        self.items = sorted(items, key=lambda t: t[0])

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx: int):
        cid, full_file = self.items[idx]
        with open(full_file, "rb") as f:
            G = pickle.load(f)
        data = from_networkx(G)  
        data.x = None           
        data.y = self.cid_to_y[cid].unsqueeze(0) 
        data.cid = torch.tensor([cid], dtype=torch.int64)
        return data

def make_splits(dataset, train=0.8, val=0.1, seed=42):
    n = len(dataset)
    n_train = int(n * train)
    n_val = int(n * val)
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=g)
