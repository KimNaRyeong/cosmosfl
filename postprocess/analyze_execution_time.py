import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def file2bug(json_file):
    if not json_file.endswith(".json"):
        return None
    try:
        return os.path.basename(json_file).removeprefix('XFL-').removesuffix('.json')
    except:
        return None

def analyze_execution_time(result_dirs, project=None):
    execution_times = {}
    timeout_count = 0
    for result_dir in result_dirs:
        file_iterator = sorted(os.listdir(result_dir))
        print(f"Processing {result_dir}...")
        for fname in file_iterator:
            bug_name = file2bug(fname)
            if bug_name is None:
                continue            
            if project and not bug_name.startswith(project):
                continue

            fpath = os.path.join(result_dir, fname)
            with open(fpath, 'r') as f:
                autofl_data = json.load(f)
            if bug_name not in execution_times:
                execution_times[bug_name] = []

            execution_time = autofl_data["time_taken"]
            if execution_time > 600:
                timeout_count += 1
            execution_times[bug_name].append(execution_time)
            
    return execution_times, timeout_count

def plot_boxes_for_each_bug(data, path):
    def shorten_labels(bug_list):
        shortened_labels = []
        previous_category = ""
        for bug in bug_list:
            category, _ = bug.split('_')
            if category == previous_category:
                shortened_labels.append('')
            else:
                shortened_labels.append(category)
                previous_category = category
        return shortened_labels

    sorted_bugs = sorted(data.keys())
    shortened_labels = shorten_labels(sorted_bugs)

    data_list = [(bug, time) for bug, times in data.items() for time in times]
    bugs, times = zip(*data_list)

    plt.figure(figsize=(15, 6))
    sns.boxplot(x=bugs, y=times, hue=bugs, palette="pastel", flierprops=dict(marker='x', markerfacecolor='gray', markersize=5))
    plt.xticks(ticks=range(len(sorted_bugs)), labels=shortened_labels, rotation=90)
    plt.yticks(fontsize=12)
    plt.title('Execution Times for Each Bug')
    plt.ylabel('Execution Time (s)')
    plt.tight_layout()
    plt.savefig(path)

def plot_boxes_for_each_category(data, path):
    data_list = [(bug.split('_')[0], time) for bug, times in data.items() for time in times]
    bugs, times = zip(*data_list)

    plt.figure(figsize=(10, 8))
    sns.boxplot(x=bugs, y=times, hue=bugs, palette="pastel", flierprops=dict(marker='x', markerfacecolor='gray', markersize=5))    
    plt.yscale('log')
    plt.yticks(fontsize=12)
    plt.title('Execution Times for Each Category', fontsize=16)
    plt.ylabel('Execution Time (s)', fontsize=12)
    plt.tight_layout()
    plt.savefig(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_dirs', nargs="+", type=str)
    parser.add_argument('--output', '-o', type=str, default="execution_time")
    parser.add_argument('--project', '-p', type=str, default=None)
    args = parser.parse_args()

    data, timeouts = analyze_execution_time(args.result_dirs, args.project)
    plot_boxes_for_each_category(data, f'{args.output}_per_category.png')
    plot_boxes_for_each_bug(data, f'{args.output}_per_bug.png')
    data['total_time'] = sum([sum(v) for v in data.values()]) 
    print(f'Total time: {data["total_time"]}')

    data['timeout_count'] = timeouts
    with open(f'{args.output}.json', "w") as f:
        json.dump(data, f, indent=4)