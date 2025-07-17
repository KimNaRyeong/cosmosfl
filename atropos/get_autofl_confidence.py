import argparse, json
from sklearn.metrics import accuracy_score, roc_auc_score

def get_accuracy(model, run_count, sample_size):
    all_labels = []
    all_preds = []
    all_autofl_confidences = []
    for i in range(1, sample_size+1):
        if model == 'equal_weight':
            model = 'accat1_de'
        fl_results_file = f'../autofl/weighted_fl_results/test/{model}/equal_R{run_count}_{sample_size}.json'
        # fl_results_file = f'../autofl/weighted_fl_results/test/equal_R{run_count}_{sample_size}.json'
        with open(fl_results_file, 'r') as f:
            results = json.load(f)
        bug_list = results['ranks'].keys()
        for bug in bug_list:
            all_labels.append(1 if results['ranks'][bug] == 1 else 0)
            all_preds.append(1 if results['autofl_confidence'][bug] >= 0.5 else 0)
            all_autofl_confidences.append(results['autofl_confidence'][bug])
    
    accuracy = accuracy_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_autofl_confidences)

    return accuracy, roc_auc

def main(model, run_count, sample_size):
    accuracy, roc_auc = get_accuracy(model, run_count, sample_size)
    result_file = f'./results/one_hot/{model}/results_gcn_R{run_count}_{sample_size}files.txt'

    acc_new_line = f'AutoFL-Confidence accuracy: {accuracy:.4f}\n'
    auc_new_line = f'AutoFL-Confidence roc-auc: {roc_auc:.4f}\n'
    with open(result_file, 'r') as rf:
        results = rf.read()
    
    with open(result_file, 'w') as rf:
        rf.write(acc_new_line + auc_new_line + results)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', '-m', type=str, default = 'llama3')
    parser.add_argument('--run_count', '-r', type=int, default=10)
    parser.add_argument('--sample_size', '-n', type=int, default=1)
    args = parser.parse_args()

    main(args.model, args.run_count, args.sample_size)