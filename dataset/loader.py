
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch

class IrisDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = torch.tensor(df.drop("species", axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(pd.factorize(df["species"])[0], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloader(config):
    dataset = IrisDataset(config["dataset"]["csv_path"])
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
