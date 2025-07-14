import torch
import torch.nn as nn
import torch.optim as optim
from utils.config import load_config
from utils.metrics import log_metrics
from datasets.loader import get_dataloader
from models.simple_model import SimpleModel
import argparse

def train(config):
    dataloader = get_dataloader(config)
    model = SimpleModel(
        config['model']['input_dim'],
        config['model']['hidden_dim'],
        config['model']['output_dim']
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    for epoch in range(config['training']['epochs']):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.4f}")

    log_metrics(
        {"final_loss": total_loss, "accuracy": acc},
        config['output']['metrics_path']
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)
