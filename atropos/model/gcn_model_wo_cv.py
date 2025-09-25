import torch.nn as nn
import os
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
    
def data_load(model, repetition, num_files, ks):
    train_dataset_S, test_dataset_S = dict(), dict()
    train_dataset_F, test_dataset_F = dict(), dict()
    train_dataset_FA, test_dataset_FA = dict(), dict()

    for k in ks:
        base = f"/home/kimnal0/cosmosfl/atropos/data/{model}/R{repetition}_{num_files}files/{k}"
        train_data = torch.load(f"{base}/train_gcn_dataset.pth", weights_only=False)
        test_data = torch.load(f"{base}/test_gcn_dataset.pth", weights_only=False)

        train_dataset_S[k], train_dataset_F[k], train_dataset_FA[k] = train_data["dataset_S"], train_data["dataset_F"], train_data["dataset_FA"]
        test_dataset_S[k], test_dataset_F[k], test_dataset_FA[k] = test_data["dataset_S"], test_data["dataset_F"], test_data["dataset_FA"]
    
    print("Datasets loaded successfully")

    return (train_dataset_S, test_dataset_S), (train_dataset_F, test_dataset_F), (train_dataset_FA, test_dataset_FA)


def print_metadata(dataset, ks, dataset_name):
    print(f"About {dataset_name}")
    print(f"Data size: {len(dataset[1])}")

    for k in sorted(ks):
        print(f"------------{k}------------")
        print(f".   Size: {len(dataset[k])}")
        print(f".   x shape: {dataset[k][0].x.shape}")


def get_baseline_acc(dataset):
    num_total = len(dataset[1])
    num_true = 0
    for data in dataset[1]:
        if data.y:
            num_true += 1
    baseline_acc = num_true / num_total
    return baseline_acc

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
    



def train_one_epoch(model, optimizer, criterion, loader, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        if out.dim() == 2 and out.size(1) == 1:
            out = out.view(-1)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

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

@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for data in loader:
        data = data.to(device)
        out = model(data).squeeze()
        pred = (torch.sigmoid(out) >= 0.5).int()
        correct += int((pred == data.y).sum())
        total += data.y.numel()
    return correct / total if total > 0 else 0.0

def train_and_test_model(train_dataset, test_dataset, criterion, output_dim, lr, batch_size, hidden_dim, dropout_p, num_layer, num_epochs, ks, result_file, device, llm_model, repetition, num_files, dataset_name):
    print(f"Training and testing with {dataset_name}")
    with open(result_file, "a+") as rf:
        rf.write(f"{dataset_name.split('_')[-1]}\n")
    
    for k in sorted(ks):
        with open(result_file, "a+") as rf:
            rf.write(f'k={k}\n')
        
        ckpt_base = f'../trained_model/{llm_model}/R{repetition}_{num_files}files/{k}'
        print(f"==================For {k}=======================")
        
        input_dim = train_dataset[k][0].x.shape[1]
        
        train_loader = DataLoader(train_dataset[k], batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset[k], batch_size=batch_size, shuffle=False)

        model = GCN(input_dim, hidden_dim, output_dim, dropout_p, num_layer).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_acc = -1.0
        best_acc_epoch = -1

        train_accs, test_accs = [], []

        graph_dir = f'/home/kimnal0/cosmosfl/atropos/results/graphs/one_hot/{llm_model}/R{repetition}_{num_files}files'

        for epoch in tqdm(range(num_epochs)):
            loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
            train_acc = eval_accuracy(model, train_loader, device)
            test_acc = eval_accuracy(model, test_loader, device)

            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                best_acc_epoch = epoch
                
                acc_ckpt = os.path.join(ckpt_base, 'best_acc.pt')
                save_checkpoint(
                    model, acc_ckpt,
                    meta = {
                        "metric": "test_acc",
                        "value": float(best_acc),
                        "epoch": int(best_acc_epoch),
                        "k": int(k),
                        "hparams": {
                            "input_dim": int(input_dim),
                            "hidden_dim": int(hidden_dim),
                            "output_dim": int(output_dim),
                            "dropout_p": float(dropout_p),
                            "num_layer": int(num_layer),
                            "lr": float(lr),
                            "batch_size": int(batch_size),
                    }
                    }
                )
                print(f"[k={k}] New checkpoint saved! = Accuracy: {best_acc:.4f} (epoch {best_acc_epoch+1})")

                # ROC-AUC curve
                fpr, tpr, auc = test_with_auc(model, test_loader, device)
                roc_auc_graph_path = os.path.join(graph_dir, f"{k}k_roc_auc_wo_cv.png")
                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC_AUC={auc:.4f}")
                plt.plot([0, 1], [0, 1], linestyle='--')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve (Best Test Acc Epoch)")
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(roc_auc_graph_path)
                plt.close()

                precision = evaluate_with_fixed_threshold_precision(model, test_loader, device)
                recall = evaluate_with_fixed_threshold_recall(model, test_loader, device)
                npv = evaluate_with_npv(model, test_loader, device)
                specificity = evaluate_with_specificity(model, test_loader, device)

        # Accuracy curve
        acc_graph_path = os.path.join(graph_dir, f"{k}k_acc_wo_cv.png")
        plt.figure()
        epochs = np.arange(1, num_epochs+1)
        plt.plot(epochs, train_accs, label="Train Acc")
        plt.plot(epochs, test_accs, label="Test Acc")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title("Train/Test Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(acc_graph_path)
        plt.close()

        print(f"\n===========for {k}===========")
        print(f"Best Test Acc: {best_acc:.4f}")

        with open(result_file, "a+") as rf:
            rf.write(f"Epoch = {best_acc_epoch}\n")
            rf.write(f"Best accuracy: {best_acc:.4f}\n")
            rf.write(f"Best roc_auc: {auc:.4f}\n")
            rf.write(f"Best precision: {precision:.4f}\n")
            rf.write(f"Best recall: {precision:.4f}\n")
            rf.write(f"Best npv: {npv:.4f}\n")
            rf.write(f"Best specificify: {specificity:.4f}\n")
        


            



        # for fold, (train_idx, test_idx) in tqdm(enumerate(splits)):
        #     train_dataset = [dataset[k][i] for i in train_idx]
        #     test_dataset = [dataset[k][i] for i in test_idx]
        #     train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        #     test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

        #     model = GCN(input_dim, hidden_dim, output_dim, dropout_p, num_layer).to(device)
        #     optimizer = torch.optim.Adam(model.parameters(), lr = lr)


        #     for epoch in range(num_epochs):
        #         loss = train(model, optimizer, criterion, train_loader,device)
        #         train_acc = test(model, train_loader, device)
        #         test_acc = test(model, test_loader, device)
        #         fpr, tpr, _ = test_with_auc(model, test_loader, device)
        #         precision = evaluate_with_fixed_threshold_precision(model, test_loader, device)
        #         recall = evaluate_with_fixed_threshold_recall(model, test_loader, device)
        #         npv = evaluate_with_npv(model, test_loader, device)
        #         specificity = evaluate_with_specificity(model, test_loader, device)

        #         train_accs[epoch] += train_acc
        #         test_accs[epoch] += test_acc
        #         precisions[epoch] += precision
        #         recalls[epoch] += recall
        #         npvs[epoch] += npv
        #         specificities[epoch] += specificity

        #         epoch_fprs[epoch].append(fpr)
        #         epoch_tprs[epoch].append(np.interp(mean_fpr, fpr, tpr))
        #         epoch_tprs[epoch][-1][0]= 0.0

        #         if test_acc > best_acc:
        #             best_acc = test_acc
        #             best_acc_fold = fold
        #             best_acc_epoch = epoch
        #             if split:
        #                 acc_ckpt = os.path.join(ckpt_base, f'train_best_acc.pt')
        #             else:    
        #                 acc_ckpt = os.path.join(ckpt_base, f'best_acc.pt')
        #             save_checkpoint(
        #                 model,
        #                 acc_ckpt,
        #                 meta={
        #                     "metric": "test_acc",
        #                     "value": float(best_acc),
        #                     "fold": int(best_acc_fold),
        #                     "epoch": int(best_acc_epoch),
        #                     "k": int(k),
        #                     "fold": int(fold),
        #                     "hparams": {
        #                         "input_dim": int(input_dim),
        #                         "hidden_dim": int(hidden_dim),
        #                         "output_dim": int(output_dim),
        #                         "dropout_p": float(dropout_p),
        #                         "num_layer": int(num_layer),
        #                         "lr": float(lr),
        #                         "batch_size": int(batch_size)
        #                     }
        #                 }
        #             )
        #             print(f"New checkpoint is saved! - Accuracy: {best_acc}")

        # mean_train_accs = [acc / K for acc in train_accs]
        # mean_test_accs = [acc / K for acc in test_accs]
        # mean_precisions = [pre / K for pre in precisions]
        # mean_recalls = [recall / K for recall in recalls]
        # mean_npvs = [npv / K for npv in npvs]
        # mean_specificities = [spec / K for spec in specificities]


        # best_mean_accuracy = max(mean_test_accs)
        # best_epoch = mean_test_accs.index(best_mean_accuracy)
        # mean_tpr = np.mean(epoch_tprs[best_epoch], axis=0)
        # std_tpr = np.std(epoch_tprs[best_epoch], axis=0)
        # mean_tpr[-1] = 1.0
        # mean_roc_auc = auc(mean_fpr, mean_tpr)

        # # Plot accuracy graphs over epochs
        # graph_dir = f'/home/kimnal0/cosmosfl/atropos/results/graphs/one_hot/{llm_model}/R{repetition}_{num_files}files'
        # if not os.path.exists(graph_dir):
        #     os.makedirs(graph_dir)

        # if split:
        #     acc_graph_path = os.path.join(graph_dir, f"train_{k}k_acc.png")
        # else:
        #     acc_graph_path = os.path.join(graph_dir, f"{k}k_acc.png")
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, num_epochs + 1), mean_train_accs, label='Train Accuracy')
        # plt.plot(range(1, num_epochs + 1), mean_test_accs, label='Test Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.title(f'Accuracy per Epoch for F+A embedding (k={k})')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(acc_graph_path)

        # roc_auc_graph_path = os.path.join(graph_dir, f"{k}k_roc_auc.png")
        # plt.figure(figsize=(4.5, 4))
        # plt.plot(mean_fpr, mean_tpr, label=f'ROC Curve (AUC = {mean_roc_auc:.4f})', color='blue', linewidth=2)
        # plt.fill_between(mean_fpr, np.maximum(mean_tpr-std_tpr, 0), np.minimum(mean_tpr+std_tpr, 1), color = 'blue', alpha=0.2, label = 'Std. Dev.')
        # plt.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=1.5, label='Random Guess')
        # plt.xlabel('False Positive Rate (FPR)', fontsize=14)
        # plt.ylabel('True Positive Rate (TPR)', fontsize=14)
        # plt.title(f'Mean ROC Curve for GCN Model (F+A, k={k})', fontsize=16)
        # plt.legend(loc='lower right', fontsize=12)
        # plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # plt.tight_layout()
        # plt.savefig(roc_auc_graph_path)
        
        # print(f"Epoch = {best_epoch}")
        # print(f"Best_mean_accuracy: {mean_test_accs[best_epoch]:.4f}")
        # print(f"Best mean_roc_auc: {mean_roc_auc:.4f}")
        # print(f"Best mean_precision: {mean_precisions[best_epoch]:.4f}")
        # print(f"Best mean_recall: {mean_recalls[best_epoch]:.4f}")
        # print(f"Best mean_npv: {mean_npvs[best_epoch]:.4f}")
        # print(f"Best mean_specificity: {mean_specificities[best_epoch]:.4f}")
        # print('-------------------------------------------------------------------')
        # with open(result_file, "a+") as rf:
        #     rf.write(f"Epoch = {best_epoch}\n")
        #     rf.write(f"Best_mean_accuracy: {best_mean_accuracy:.4f}\n")
        #     rf.write(f"Best mean_roc_auc: {mean_roc_auc:.4f}\n")
        #     rf.write(f"Best mean_precision: {mean_precisions[best_epoch]:.4f}\n")
        #     rf.write(f"Best mean_recall: {mean_recalls[best_epoch]:.4f}\n")
        #     rf.write(f"Best mean_npv: {mean_npvs[best_epoch]:.4f}\n")
        #     rf.write(f"Best mean_specificity: {mean_specificities[best_epoch]:.4f}\n")
    
def save_checkpoint(model, path, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if meta:
        payload.update(meta)
    torch.save(payload, path)



def main(model, repetition, num_files):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ks = list(range(1, 13))

    (train_S, test_S), (train_F, test_F), (train_FA, test_FA) = data_load(model, repetition, num_files, ks)
    # get_baseline_acc(train_FA, result_file)
    print_metadata(train_FA, ks, "train_FA")
    print_metadata(test_FA, ks, "test_FA")

    result_dir = f"/home/kimnal0/cosmosfl/atropos/results/one_hot/{model}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_file = os.path.join(result_dir, f"results_gcn_R{repetition}_{num_files}files_wo_cv.txt")

    if os.path.exists(result_file):
        os.remove(result_file)
        print(f"{result_file} is removed")

    criterion = nn.BCEWithLogitsLoss()
    output_dim = 1
    lr = 0.001
    batch_size = 32
    hidden_dim = 32
    dropout_p = 0.8
    num_layer = 2
    num_epochs = 150

    train_FA_baseline_acc = get_baseline_acc(train_FA)
    test_FA_baseline_acc = get_baseline_acc(test_FA)
    print(f"Train FA Baseline accuracy: {train_FA_baseline_acc:.4f}")
    print(f"Test FA Baseline accuracy: {test_FA_baseline_acc:.4f}")

    with open(result_file, 'a+') as rf:
        rf.write(f'Train FA baseline accuracy: {train_FA_baseline_acc:.4f}\n')
        rf.write(f'Test FA baseline accuracy: {test_FA_baseline_acc:.4f}\n')
    # train_and_test_model(dataset_S, criterion, output_dim, K, kf, lr, batch_size, hidden_dim, dropout_p, num_layer, num_epochs, ks, result_file, device, model, repetition, num_files, "dataset_S", split)
    # train_and_test_model(dataset_F, criterion, output_dim, K, kf, lr, batch_size, hidden_dim, dropout_p, num_layer, num_epochs, ks, result_file, device, model, repetition, num_files, "dataset_F", split)
    train_and_test_model(train_FA, test_FA, criterion, output_dim, lr, batch_size, hidden_dim, dropout_p, num_layer, num_epochs, ks, result_file, device, model, repetition, num_files, "dataset_FA")




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