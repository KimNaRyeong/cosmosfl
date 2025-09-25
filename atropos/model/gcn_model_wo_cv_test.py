import torch.nn as nn
import os
import json
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def data_load_test_only(model, repetition, num_files, ks):
    test_dataset_S, test_dataset_F, test_dataset_FA = dict(), dict(), dict()

    for k in ks:
        base = f"/home/kimnal0/cosmosfl/atropos/data/{model}/R{repetition}_{num_files}files/{k}"
        test_data = torch.load(f"{base}/test_gcn_dataset.pth", weights_only=False)

        test_dataset_S[k], test_dataset_F[k], test_dataset_FA[k] = test_data["dataset_S"], test_data["dataset_F"], test_data["dataset_FA"]
    
    print("Datasets loaded successfully")

    return test_dataset_S, test_dataset_F, test_dataset_FA

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])
        self.conv_out = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout_p = dropout_p

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p = self.dropout_p, training=self.training)
        x = self.conv_out(x, edge_index, edge_weight)
        x = global_mean_pool(x, batch) 
        x = self.fc(x)
        return x

def load_gcn_from_ckpt(ckpt_path, device):
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = payload.get("hparams", {})
    input_dim = int(meta["input_dim"])
    hidden_dim = int(meta["hidden_dim"])
    output_dim = int(meta["output_dim"])
    dropout_p = float(meta["dropout_p"])
    num_layer = int(meta["num_layer"])

    model = GCN(input_dim, hidden_dim, output_dim, dropout_p, num_layer).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model

@torch.no_grad()
def dump_predictions_from_checkpoints(test_dataset, llm_model, repetition, num_files, ks, device, out_json_path):
    agg = {}
    for k in sorted(ks):
        samples = test_dataset.get(k, [])
        if len(samples) == 0:
            print(f"[Warn] empty test set at k={k}")
            continue
        
        ckpt_path = f'../trained_model/{llm_model}/R{repetition}_{num_files}files/{k}/best_acc.pt'
        input_dim = samples[0].x.shape[1]
        model = load_gcn_from_ckpt(ckpt_path, device)

        print(f"[k={k}] predicting {len(samples)} samples...")
        for data in tqdm(samples, leave=False):
            bug_name = data.bug_name
            file_idx = int(data.file_idx)
            y = int(data.y.item())

            d = data.to(device)
            logits = model(d).view(-1)
            prob = torch.sigmoid(logits)[0].item()
            pred_bin = int(prob>=0.5)

            if file_idx not in agg:
                agg[file_idx] = {}
            if bug_name not in agg[file_idx]:
                agg[file_idx][bug_name] = {"gt_label": y, "prediction": {}}
            agg[file_idx][bug_name]["prediction"][k] = pred_bin
    
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(agg, f, indent = 2)
    print(f"[ok] prediction dump saved to: {out_json_path}")
    return agg
        

def main(model, repetition, num_files):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ks = list(range(1, 13))

    test_S, test_F, test_FA = data_load_test_only(model, repetition, num_files, ks)

    out_json = f"../results/one_hot/{model}/R{repetition}_{num_files}files/predictions_FA.json"
    dump_predictions_from_checkpoints(test_FA, model, repetition, num_files, ks, device, out_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='llama3')
    parser.add_argument('-r', '--repetition', default=1, type=int)
    parser.add_argument('-n', '--num_files', default=1, type=int)

    args = parser.parse_args()
    assert args.model in ['llama3', 'llama3.1', 'mistral-nemo', 'qwen2.5-coder', 'equal_weight']
    assert args.repetition in range(1, 25)
    assert args.num_files in range(1, 21)

    main(args.model, args.repetition, args.num_files)