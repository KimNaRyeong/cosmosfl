import torch.nn as nn
import os
import torch
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split


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
    
def data_load(ks):
    dataset_S = dict()
    dataset_F = dict()
    dataset_FA = dict()

    for k in ks:
        loaded_data = torch.load(f"/home/kimnal0/cosmosfl/atropos/data/R10_N3/{k}/gcn_dataset.pth", weights_only=False)

        dataset_S[k] = loaded_data["dataset_S"]
        dataset_F[k] = loaded_data["dataset_F"]
        dataset_FA[k] = loaded_data["dataset_FA"]
    
    print("Datasets loaded successfully")

    return dataset_S, dataset_F, dataset_FA


def print_metadata(dataset, ks, dataset_name):
    print(f"About {dataset_name}")
    print(f"Data size: {len(dataset[1])}")
    print(f"ks: {list(dataset.keys())}")

    for k in sorted(ks):
        print(f"------------{k}------------")
        print(f".   Size: {len(dataset[k])}")
        print(f".   x shape: {dataset[k][0].x.shape}")


def get_baseline_acc(dataset, result_file):
    num_total = len(dataset[1])
    num_true = 0
    for data in dataset[1]:
        if data.y:
            num_true += 1
    baseline_acc = num_true / num_total
    print(f"Baseline accuracy: {baseline_acc}")

    with open(result_file, 'a+') as rf:
        rf.write(f'baseline accuracy: {baseline_acc:.4f}\n')


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
    



def train(model, optimizer, criterion, train_loader, device):
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        if out.dim() == 2 and out.size(1) == 1:
            out = out.view(-1)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(train_loader))

def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data).squeeze()
        pred = (torch.sigmoid(out) >= 0.5).int()
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


def test_with_auc(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.view(-1)

            preds = torch.sigmoid(out).cpu().numpy()
            
            labels = data.y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    return fpr, tpr, auc

def evaluate_with_fixed_threshold_precision(model, loader, device):
    threshold=0.5
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.view(-1)
                
            pred = (torch.sigmoid(out) >= threshold).int()

            if pred.ndim == 0:
                pred = torch.tensor(pred.item())
            labels = data.y.cpu().numpy()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels)

    precision = precision_score(all_labels, all_preds, zero_division=0)
    
    return precision

def evaluate_with_fixed_threshold_recall(model, loader, device):
    threshold = 0.5
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.view(-1)
            pred = (torch.sigmoid(out) >= threshold).int()
            labels = data.y.cpu().numpy()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels)
    
    recall = recall_score(all_labels, all_preds, zero_division=0)
    
    return recall

def evaluate_with_npv(model, loader, device):
    threshold = 0.5
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.view(-1)
            pred = (torch.sigmoid(out) >= threshold).int()
            labels = data.y.cpu().numpy()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels)
    
    # Confusion matrix에서 True Negative와 False Negative 값을 추출
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    # Negative Predictive Value (NPV) 계산
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 나누는 값이 0이 아닐 경우에만 계산
    
    return npv

def evaluate_with_specificity(model, loader, device):
    threshold = 0.5
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            if out.dim() == 2 and out.size(1) == 1:
                out = out.view(-1)
            pred = (torch.sigmoid(out) >= threshold).int()
            labels = data.y.cpu().numpy()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels)
    
    # Confusion matrix에서 True Negative와 False Positive 값을 추출
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    # Specificity (True Negative Rate) 계산
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 나누는 값이 0이 아닐 경우에만 계산
    
    return specificity

def train_and_test_model(dataset, criterion, output_dim, lr, batch_size, hidden_dim, dropout_p, num_layer, num_epochs, ks, result_file, device, dataset_name):
    print(f"Training and testing with {dataset_name}")
    with open(result_file, "a+") as rf:
        rf.write(f"{dataset_name.split('_')[-1]}\n")
    
    for k in sorted(ks):
        with open(result_file, "a+") as rf:
            rf.write(f'k={k}\n')
        
        ckpt_base = f'../trained_model/R10_N3/{k}'
        print(f"==================For {k}=======================")
        
        input_dim = dataset[k][0].x.shape[1]

        train_dataset, test_dataset = train_test_split(dataset[k], test_size=0.2, random_state=42)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = GCN(input_dim, hidden_dim, output_dim, dropout_p, num_layer).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_acc = -1.0
        best_acc_epoch = -1

        train_accs, test_accs = [], []

        graph_dir = f'/home/kimnal0/cosmosfl/atropos/result/graphs/R10_N3'
        os.makedirs(graph_dir, exist_ok=True)

        for epoch in tqdm(range(num_epochs)):
            loss = train(model, optimizer, criterion, train_loader, device)
            train_acc = test(model, train_loader, device)
            test_acc = test(model, test_loader, device)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                best_acc_epoch = epoch

                acc_ckpt = os.path.join(ckpt_base, 'best_acc.pt')
                save_checkpoint(
                    model, acc_ckpt,
                    meta={
                        "metric": "test_acc",
                        "value": float(best_acc),
                        "epoch": int(best_acc_epoch),
                        "k": int(k),
                        "hparams": {
                            'input_dim': int(input_dim),
                            'hidden_dim': int(hidden_dim),
                            'output_dim': int(output_dim),
                            'dropout_p': float(dropout_p),
                            'num_layer': int(num_layer),
                            'lr': float(lr),
                            'batch_size': int(batch_size)
                        }
                    }
                )

                print(f"[k={k}] New checkpoint saved! = Accuracy: {best_acc:.4f} (epoch {best_acc_epoch+1})")

                fpr, tpr, auc = test_with_auc(model, test_loader, device)
                roc_auc_graph_path = os.path.join(graph_dir, f"{k}k_roc_auc_wo_cv.png")
                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC_AUC={auc:.4f}")
                plt.plot([0, 1], [1, 0], linestyle='--')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve (Best Test Acc Epoch)")
                plt.legend(loc = 'lower right')
                plt.tight_layout()
                plt.savefig(roc_auc_graph_path)
                plt.close()

                precision = evaluate_with_fixed_threshold_precision(model, test_loader, device)
                recall = evaluate_with_fixed_threshold_recall(model, test_loader, device)
                npv = evaluate_with_npv(model, test_loader, device)
                specificity = evaluate_with_specificity(model, test_loader, device)
        
        acc_graph_path = os.path.join(graph_dir, f"{k}k_acc_wo_cv.png")
        plt.figure()
        epochs = np.arange(1, num_epochs+1)
        plt.plot(epochs, train_accs, label = 'Train Acc')
        plt.plot(epochs, test_accs, label='Test Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Train/Test Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(acc_graph_path)
        plt.close()

        print(f"\n==============for {k}===============")
        print(f"Best Test Acc: {best_acc:.4f}")

        with open(result_file, "a+") as rf:
            rf.write(f"Epoch = {best_acc_epoch}\n")
            rf.write(f"Best accuracy: {best_acc:.4f}\n")
            rf.write(f"Best roc_auc: {auc:.4f}\n")
            rf.write(f"Best precision: {precision:.4f}\n")
            rf.write(f"Best recall: {recall:.4f}\n")
            rf.write(f"Best npv: {npv:.4f}\n")
            rf.write(f"Best specificify: {specificity:.4f}\n")

    
def save_checkpoint(model, path, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if meta:
        payload.update(meta)
    torch.save(payload, path)



def main(model):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ks = list(range(1, 13))

    dataset_S, dataset_F, dataset_FA = data_load(ks)

    result_dir = f"/home/kimnal0/cosmosfl/atropos/results/one_hot/R10_N3"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_file = os.path.join(result_dir, f"results_gcn_R10_N3.txt")

    if os.path.exists(result_file):
        os.remove(result_file)
        print(f"{result_file} is removed")
    
    get_baseline_acc(dataset_S, result_file)
    print_metadata(dataset_S, ks, "dataset_S")
    print_metadata(dataset_F, ks, "dataset_F")
    print_metadata(dataset_FA, ks, "dataset_FA")

    criterion = nn.BCEWithLogitsLoss()
    output_dim = 1
    lr = 0.001
    batch_size = 32
    hidden_dim = 32
    dropout_p = 0.8
    num_layer = 2
    num_epochs = 150

    train_and_test_model(dataset_S, criterion, output_dim, lr, batch_size, hidden_dim, dropout_p, num_layer, num_epochs, ks, result_file, device, "dataset_S")
    train_and_test_model(dataset_F, criterion, output_dim, lr, batch_size, hidden_dim, dropout_p, num_layer, num_epochs, ks, result_file, device, "dataset_F")
    train_and_test_model(dataset_FA, criterion, output_dim, lr, batch_size, hidden_dim, dropout_p, num_layer, num_epochs, ks, result_file, device, "dataset_FA")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='llama3')
    args = parser.parse_args()
    assert args.model in ['llama3', 'llama3.1', 'mistral-nemo', 'qwen2.5-coder', 'equal_weight']

    main(args.model)