import csv
import argparse

def main(args):
    input_result_file = args.input_file

    with open(input_result_file, 'r') as f:
        results = f.readlines()
    k=0
    result_dict = {}
    for l in results:
        l = l.strip()
        if l=='S':
            embedding = 'S'
            result_dict[embedding] = {}
        elif l=='F':
            embedding = 'F'
            result_dict[embedding] = {}
        elif l=='FA':
            embedding = 'FA'
            result_dict[embedding] = {}
        elif l.startswith('k='):
            k=l[2:]
            result_dict[embedding][k] = {}
        elif l.startswith('Best_mean_accuracy:'):
            result_dict[embedding][k]['acc'] = l.split()[-1]
        elif l.startswith('Best mean_roc_auc:'):
            result_dict[embedding][k]['roc_auc'] = l.split()[-1]
        elif l.startswith('Best mean_precision:'):
            result_dict[embedding][k]['precision'] = l.split()[-1]
        elif l.startswith('Best mean_recall:'):
            result_dict[embedding][k]['recall'] = l.split()[-1]
        elif l.startswith('Best mean_npv:'):
            result_dict[embedding][k]['npv'] = l.split()[-1]
        elif l.startswith('Best mean_specificity:'):
            result_dict[embedding][k]['spec'] = l.split()[-1]
        print("-----------------")
        print(l)
        print(embedding)
        print(k)
        print(result_dict)

    print(result_dict)

    if input_result_file.endswith('.txt'):
        csv_file = input_result_file[:-3] + '.csv'

    with open(csv_file, 'w') as f:
        writer = csv.writer(f)

        writer.writerow(input_result_file)

        if 'S' in result_dict.keys():
            writer.writerow('S')
            for k, values in result_dict['S'].items():
                row = [values[key] for key in ['acc', 'roc_auc', 'precision', 'recall', 'npv', 'spec']]
                writer.writerow(row)

        if 'F' in result_dict.keys():
            writer.writerow('F')
            for k, values in result_dict['F'].items():
                row = [values[key] for key in ['acc', 'roc_auc', 'precision', 'recall', 'npv', 'spec']]
                writer.writerow(row)
            
        writer.writerow('FA')

        for k, values in result_dict['FA'].items():
            row = [values[key] for key in ['acc', 'roc_auc', 'precision', 'recall', 'npv', 'spec']]
            writer.writerow(row)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type = str, default='./lstm_two_full.txt', help='txt file for result')

    args = parser.parse_args()
    main(args)