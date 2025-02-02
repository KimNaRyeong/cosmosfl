import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def analyze_cost(result_dirs):
    costs = {}
    for result_dir in result_dirs:
        file_iterator = sorted(os.listdir(result_dir))
        print(f"Processing {result_dir}...")

        model_name = result_dir.split('/')[-1]
        if model_name not in costs:
            costs[model_name] = {'total_energy': list(), 'num_of_queries':list(), 'time_taken': list(), 'avg_power': list()}
        for fname in file_iterator:
            fpath = os.path.join(result_dir, fname)
            with open(fpath, 'r') as f:
                autofl_data = json.load(f)
            num_of_queries = sum([1 for msg in autofl_data["messages"] if msg["role"] == "assistant"]) - 1 # exclude step 0(class_cov)
            if not num_of_queries:
                print(f'WARNING: Zero interactions recorded for {result_dir}/{fname}')
            else:
                costs[model_name]['total_energy'].append(sum(autofl_data["total_energy"]))
                costs[model_name]['num_of_queries'].append(num_of_queries)
                costs[model_name]['time_taken'].append(autofl_data['time_taken'])
                costs[model_name]['avg_power'].append(sum(autofl_data["total_energy"]) / autofl_data['time_taken']) 
    return costs

def plot_boxes_for_each_model(data, path_header):
    for model, history in data.items():
        consumptions = np.array(history['total_energy']) / 1000
        nums_of_queries = history['num_of_queries']

        plt.figure(figsize=(15, 6))
        sns.boxplot(x=nums_of_queries, y=consumptions, hue=nums_of_queries, palette="flare")
        plt.yticks(fontsize=12)
        plt.yscale('log')
        plt.title(f'Energy Consumption for Interaction Count ({model})')
        plt.xlabel('Number of Queries')
        plt.ylabel('Energy Consumption (kJ)')
        plt.tight_layout()
        plt.savefig(f'{path_header}_{model}.png')

def plot_boxes_for_multi_models(data, path_header):
    models = list()
    original_consumptions = list()
    stepwise_consumptions = list()
    consumptions_per_second = list()
    time_taken = list()
    
    for model, history in data.items():
        consumptions = history['total_energy']
        nums_of_queries = history['num_of_queries']
        times_taken = history['time_taken']
        models.extend([model] * len(consumptions))
        original_consumptions.extend(consumptions)
        stepwise_consumptions.extend([consumption / step for consumption, step in zip(consumptions, nums_of_queries)])
        consumptions_per_second.extend([consumption / time_taken for consumption, time_taken in zip(consumptions, times_taken)])
        time_taken.extend(times_taken)

    colors = ['#a6611a','#dfc27d','#80cdc1','#018571']
    palette = dict(zip(set(models), colors))
        
    def save_box_plot(y, title, ylabel, output_path, is_log=True): 
        plt.figure()
        sns.boxplot(x=models, y=y, hue=models, palette=palette)
        plt.yticks(fontsize=12)
        if is_log:
            plt.yscale('log')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(output_path)
    
    save_box_plot(np.array(original_consumptions) / 1000, 'Model - Energy Consumptions', 'Energy Consumption (kJ)', f'{path_header}_original.png')
    save_box_plot(np.array(stepwise_consumptions) / 1000, 'Model - Energy Consumptions per Queries', 'Energy Consumption (kJ)', f'{path_header}_energy_per_queries.png')
    save_box_plot(consumptions_per_second, 'Model - Energy Consumptions per Second', 'Energy Consumption / Sec (Watt)', f'{path_header}_energy_per_seconds.png', is_log=False)
    save_box_plot(time_taken, 'Model - Execution Time', 'Time Taken (Sec)', f'{path_header}_execution_time.png')

def summarize_per_model(data, path_header):
    for model, val in data.items():
        df = pd.DataFrame(val) 
        data[model]['summary'] = str(df.describe())
        print(data[model]['summary'])

        plt.figure(figsize=(16, 6))
        heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, annot_kws={"size":16}, cmap='BrBG')
        heatmap.set_title(f'Correlation Heatmap ({model})', fontdict={'fontsize':18}, pad=12)
        plt.savefig(f'{path_header}_corr_{model}.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_dirs', nargs="+", type=str)
    parser.add_argument('--output', '-o', type=str, default="cost")
    args = parser.parse_args()

    data = analyze_cost(args.result_dirs)
    plot_boxes_for_multi_models(data, args.output)
    plot_boxes_for_each_model(data, args.output)
    # summarize_per_model(data, args.output)

    with open(f'{args.output}.json', "w") as f:
        json.dump(data, f, indent=4)