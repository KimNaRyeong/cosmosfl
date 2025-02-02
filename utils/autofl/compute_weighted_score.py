import json
import os
import argparse
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from time import time
from sklearn.model_selection import KFold, StratifiedKFold

from lib.repo_interface import get_repo_interface
from compute_score import *

import sys
sys.path.insert(0, '..')
from utils.optimization_strategies import *

NUM_DATAFRAME_HEADER_COLS = 7

def compute_model_scores(result_dirs, project=None):
    json_status = {}
    score_results = {}
    model_list = []
    for result_dir in result_dirs:
        file_iterator = sorted(os.listdir(result_dir))
        file_iterator = tqdm(file_iterator)
        for fname in file_iterator:
            bug_name = file2bug(fname)
            if bug_name is None:
                continue            
            if project and not bug_name.startswith(project):
                continue

            json_status[bug_name] = json_status.get(bug_name, {"OK": [], "OtherError": [], "InvalidRequestError": []}) # status -> list
            score_results[bug_name] = score_results.get(bug_name, {})   # method -> model -> score info
            status_result, score_result = json_status[bug_name], score_results[bug_name]
    
            fpath = os.path.join(result_dir, fname)
            model = result_dir.split('/')[-1]
            if model not in model_list:
                model_list.append(model)
            with open(fpath, 'r') as f:
                autofl_data = json.load(f)

            prediction = autofl_data["buggy_methods"]
            pred_status = get_prediction_status(prediction)

            status_result[pred_status] = status_result.get(pred_status, [])
            status_result[pred_status].append(fpath)

            """
            Get LLM answer
            """
            final_response = autofl_data["messages"][-1]["content"]
            pred_exprs = parse_response(final_response)

            """
            Scoring
            """

            # 1. Initialize
            ri = get_repo_interface(bug_name)

            # 2. Get mactching methods
            predicted_methods = {}
            for pred_expr in pred_exprs:
                for method in ri.get_matching_method_signatures(pred_expr):
                    predicted_methods[method] = predicted_methods.get(method, [])
                    predicted_methods[method].append(pred_expr)

            # 3. Assign scores
            # Evenly distribute the score "1" to all matching methods 
            for method in predicted_methods:
                if method not in score_result:
                    score_result[method] = {"count": 0}
                if model not in score_result[method]: 
                    score_result[method][model] = 0
                score_result[method]["count"] += 1
                score_result[method][model] += 1/len(predicted_methods)

    for bug_name in score_results:
        ri = get_repo_interface(bug_name)
        all_methods = ri.method_signatures
        buggy_methods = ri.buggy_method_signatures
        score_result = score_results[bug_name]
        num_all_runs = sum([len(json_status[bug_name][s]) for s in json_status[bug_name]])
        for method in sorted(all_methods): # lexical sort
            if method not in score_result:
                score_result[method] = {"count": 0}
            for model in model_list:
                if model not in score_result[method]: 
                    score_result[method][model] = 0
                score_result[method][model] /= num_all_runs
            score_result[method]['is_buggy'] = 1 if method in buggy_methods else 0

    return model_list, json_status, score_results

def turn_dict_into_dataframe(autofl_scores, model_list):
    flattened_data = []
    for bug, methods in autofl_scores.items():
        i = 1
        for method, data in methods.items():
            row = [i, bug, method, data['is_buggy'], data['aux_score'][0], data['aux_score'][1]]
            for model in model_list:
                row.append(data.get(model, 0.0))
            flattened_data.append(row)
            i += 1
    columns = ['i', 'bug', 'method', 'desired_score', 'aux1', 'aux2']
    columns.extend(model_list)
    df = pd.DataFrame(flattened_data, columns=columns)
    
    return df

def merge_individual_scores(result_dirs):
    scores = dict()
    for dir in result_dirs:
        model = dir.split('/')[-1]
        cache_path = f'cached_results/{dir.replace("/", "_")}.csv'
        if model not in scores:
            scores[model] = list()
        scores[model].append(pd.read_csv(cache_path))
        
    model_list = list(scores.keys())
    merged_df = scores[model_list[0]][0].copy()
    merged_df['aux2'] = 0.0
    for model in model_list:
        merged_df[model] = 0.0

    for model in model_list:
        for score in scores[model]:
            score_aligned = pd.merge(merged_df[['bug', 'method']], score)
            merged_df[model] += score_aligned[model]
            merged_df['aux2'] += score_aligned['aux2']
            merged_df['priority'] = merged_df[model_list].sum(axis=1) > 0
            merged_df = merged_df.sort_values(by=['bug', 'priority'], ascending=[True, False], ignore_index=True)
            merged_df['i'] = merged_df.groupby('bug').cumcount() + 1
    
    merged_df[model_list] /= len(result_dirs)
    aux2_mask = (merged_df[model_list] == 0).all(axis=1)
    merged_df['aux2'] *= aux2_mask
    merged_df.drop(columns=['priority'], inplace=True)

    return merged_df, model_list

def preprocess_results(result_dirs, project, aux, lang):
    if os.path.isdir('cached_results') and all([os.path.isfile(f'cached_results/{dir.replace("/", "_")}.csv') for dir in result_dirs]):
        return merge_individual_scores(result_dirs)
    else:
        model_list, json_files, autofl_scores = compute_model_scores(result_dirs, project)

        if aux:
            method_scores = add_auxiliary_scores(json_files, autofl_scores, lang, verbose=True)
        else:
            method_scores = add_auxiliary_scores(json_files, autofl_scores, lang, default_aux_score=0, verbose=True)

        return turn_dict_into_dataframe(method_scores, model_list), model_list

def apply_weight_and_evaluate(autofl_scores, model_list, weights, verbose=False):
    normalizing_factor = len(model_list) / sum(weights)
    weights = [w * normalizing_factor for w in weights]
    if verbose:
        print(f'Applying weights: {weights}')
    autofl_scores_aug = autofl_scores.copy(deep=True)
    autofl_scores_aug['weighted_sum'] = autofl_scores_aug[model_list].dot(weights)

    autofl_scores_aug.sort_values(
        by=['weighted_sum', 'aux1', 'aux2', 'i', 'method'],
        ascending=[False, False, False, True, True],
        inplace=True
    )

    autofl_scores_aug['rank'] = autofl_scores_aug.groupby('bug').cumcount() + 1

    return autofl_scores_aug[autofl_scores_aug['desired_score'] == 1].groupby('bug')['rank'].min()

def get_accuracies(rank_by_bug):
   return [len(rank_by_bug[rank_by_bug <= 1]), len(rank_by_bug[rank_by_bug <= 2]), len(rank_by_bug[rank_by_bug <= 3]), len(rank_by_bug[rank_by_bug <= 4]), len(rank_by_bug[rank_by_bug <= 5])]

def get_wef(rank_by_bug):
    return sum(rank_by_bug)

class Evaluator():
    def __init__(self, score_df, model_list):
        self.score_df = score_df
        self.model_list = model_list
    
    def evaluate(self, weight):
        ranks = apply_weight_and_evaluate(self.score_df, self.model_list, weight) 
        return get_accuracies(ranks)[0], -get_wef(ranks)

def create_evaluation_function(score_df, model_list):
    evaluator = Evaluator(score_df, model_list)
    return evaluator.evaluate

def reconstruct_dict_from_dataframe(score_df):
    data = {}
    for _, row in score_df.iterrows():
        bug = row['bug']
        method = row['method']
        if bug not in data:
            data[bug] = {}
        if method not in data[bug]:
            data[bug][method] = {}
        data[bug][method]['score'] = row['weighted_sum']
        data[bug][method]['aux_score'] = (row['aux1'], row['aux2'])
    return data

def verify_acc_with_existing_pipe(weighted_scores_df):
    method_scores = reconstruct_dict_from_dataframe(weighted_scores_df)
    method_scores = assign_rank(method_scores)
    buggy_method_ranks = get_buggy_method_ranks(method_scores, key="autofl_rank")

    summary = {"total": len(method_scores)}
    for n in range(1, 11):
        summary[f"acc@{n}"] = calculate_acc(buggy_method_ranks, key="autofl_rank", n=n)
    print(json.dumps(summary, indent=4))

def run_for_a_fold(evaluator, optimizer, validation_set, fold_size, model_list, queue):
    start = time()
    current_fold = dict()
    
    best, log, best_over_time = optimizer(evaluator)
    validation_ranks = apply_weight_and_evaluate(validation_set, model_list, best, verbose=True)
    accs = get_accuracies(validation_ranks)
        
    current_fold['best'] = best
    current_fold['best_weights_over_time'] = best_over_time
    current_fold['accs'] = accs
    current_fold['fold_size'] = fold_size
    current_fold['val_ranks'] = dict(zip(validation_ranks.index, validation_ranks.tolist()))
    current_fold['time_taken'] = time() - start 
    current_fold['log'] = log
    
    queue.append(current_fold)

def cross_validation(score_df, model_list, optimizer, k=10, stratified=False):
    cv_log = dict()
    cv_log['num_folds'] = k
    start = time()

    unique_bugs = score_df['bug'].unique()
    if stratified:
        projects = [bug_id.split('_')[0] for bug_id in unique_bugs]
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_indices = list(skf.split(unique_bugs, projects))
    else: 
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_indices = list(kf.split(unique_bugs))
    
    manager = mp.Manager()
    queue = manager.list()
    inputs = []
    for train_bug_indices, validation_bug_indices in fold_indices:
        train_set = score_df[score_df['bug'].isin(unique_bugs[train_bug_indices])]
        evaluator = create_evaluation_function(train_set, model_list)
        fold_size = len(validation_bug_indices)
        validation_set = score_df[score_df['bug'].isin(unique_bugs[validation_bug_indices])]
        inputs.append([evaluator, optimizer, validation_set, fold_size, model_list, queue])
    
    with mp.Pool(min(k, mp.cpu_count())) as p:
        p.starmap(run_for_a_fold, inputs)
    
    added_accs = [0, 0, 0, 0, 0]
    validation_ranks = dict()
    for i, fold_log in enumerate(queue):
        cv_log[i] = fold_log
        validation_ranks.update(fold_log['val_ranks'])
        added_accs = [added_accs[i] + fold_log['accs'][i] for i in range(5)]
    
    cv_log['models'] = model_list
    cv_log['total_accs'] = added_accs
    cv_log['total_time'] = time() - start
    cv_log['validation_ranks'] = validation_ranks
    
    return cv_log

def get_equal_weight(size):
    def return_equal_weight(evaluator):
        return [1] * size, '', [[1] * size] 
    return return_equal_weight

def get_correpsonding_optimizer(strategy, size):
    if strategy == 'equal':
        return get_equal_weight(size)
    elif strategy == "grid":
        return get_grid_searcher(size)
    else:
        return get_de_optimizer(size)

def get_samples(result_dirs, run_count, sample_size, max_index, sampled_indices):
    samples = []
    while len(sampled_indices) < sample_size:
        sample_index = sorted(random.sample(range(1, max_index + 1), run_count))
        if sample_index in sampled_indices:
            continue
        sampled_indices.append(sample_index)
        samples.append([dir for dir in result_dirs if any([index == int(dir.split('/')[1].split('_')[-1])for index in sample_index])])
        
    return samples

def get_existing_samples(output_dir, prefix):
    existing_files = [file for file in os.listdir(output_dir) if file.startswith(prefix)]
    sampled_dir_indices = []
    sampled_indices = []
    
    for file in existing_files:
        with open(os.path.join(output_dir, file)) as f:
            data = json.load(f)
        sampled_dirs = data['sampled_dirs']
        indices = list(set([int(file.split('/')[1].split('_')[-1]) for file in sampled_dirs]))
        sampled_dir_indices.append(indices)
        sampled_indices.append(int(file.split('_')[-1][:-5])) # trim trailing .json

    return sampled_dir_indices, sampled_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_dirs', nargs="+", type=str)
    parser.add_argument('--models', '-m', nargs="+", type=str, default='llama3')
    parser.add_argument('--output', '-o', type=str, default="scores.json")
    parser.add_argument('--project', '-p', type=str, default=None)
    parser.add_argument('--language', '-l', type=str, default="java")
    parser.add_argument('--strategy', '-s', type=str, default="de")
    parser.add_argument('--sampling', '-S', action="store_true")
    parser.add_argument('--run_count', '-R', type=int, default=2)
    parser.add_argument('--sample_size', '-N', type=int, default=20)
    parser.add_argument('--cross_validation', '-cv', action="store_true")
    parser.add_argument('--aux', '-a', action="store_true")    
    parser.add_argument('--preprocessing', action="store_true")    
    args = parser.parse_args()
    assert args.language in ["java", "python"]

    if args.preprocessing:
        models = ['llama3', 'llama3.1', 'mistral-nemo', 'qwen2.5-coder']
        filtered_dirs = [dir for dir in args.result_dirs if any([model in dir for model in models])]
        for dir in filtered_dirs:
            score_df, model_list = preprocess_results([dir], args.project, args.aux, args.language)
            with open(f'cached_results/{dir.replace("/","_")}.csv', 'w') as f:
                score_df.to_csv(f)
    elif args.sampling:
        filtered_dirs = [dir for dir in args.result_dirs if any([dir.endswith(model) for model in args.models])]
        max_index = max([int(dir.split('/')[1].split('_')[-1]) for dir in filtered_dirs])
        sampled_indices, run_indices = get_existing_samples(args.output, f'{args.strategy}_CV_R{args.run_count}_' if args.cross_validation else f'{args.strategy}_R{args.run_count}_')
        remaining_indices = list(set(range(1, args.sample_size + 1)) - set(run_indices))
        samples = get_samples(filtered_dirs, args.run_count, args.sample_size, max_index, sampled_indices)
        
        for i, sample in enumerate(samples):
            score_df, model_list = preprocess_results(sample, args.project, args.aux, args.language)    
            optimizer = get_correpsonding_optimizer(args.strategy, len(model_list))

            if args.cross_validation:
                log = cross_validation(score_df, model_list, optimizer)
                log['sampled_dirs'] = sample
            else:
                evaluator = create_evaluation_function(score_df, model_list)
                best, optimization_log, best_over_time = optimizer(evaluator)
                ranks = apply_weight_and_evaluate(score_df, model_list, best, verbose=True)
                accs = get_accuracies(ranks)
                log = dict()
                log['best'] = best
                log['best_weights_over_time'] = best_over_time
                log['accs'] = accs
                log['ranks'] = dict(zip(ranks.index, ranks.tolist()))
                log['log'] = optimization_log
                log['sampled_dirs'] = sample
            
            output_path = f'{args.output}/{args.strategy}_CV_R{args.run_count}_{remaining_indices[i]}.json' if args.cross_validation else f'{args.output}/{args.strategy}_R{args.run_count}_{remaining_indices[i]}.json'
            with open(output_path, 'w') as f:
                json.dump(log, f, indent=4)
    else:
        filtered_dirs = [dir for dir in args.result_dirs if any([dir.endswith(model) for model in args.models])]
        score_df, model_list = preprocess_results(filtered_dirs, args.project, args.aux, args.language)    

        optimizer = get_correpsonding_optimizer(args.strategy, len(model_list))

        if args.cross_validation:
            log = cross_validation(score_df, model_list, optimizer)
        else:
            evaluator = create_evaluation_function(score_df, model_list)
            best, optimization_log, best_over_time = optimizer(evaluator)
            ranks = apply_weight_and_evaluate(score_df, model_list, best, verbose=True)
            accs = get_accuracies(ranks)
            log = dict()
            log['best'] = best
            log['best_weights_over_time'] = best_over_time
            log['accs'] = accs
            log['models'] = model_list
            log['log'] = optimization_log
            log['ranks'] = dict(zip(ranks.index, ranks.tolist()))
        log['sampled_dirs'] = filtered_dirs
        output_path = f'{args.output}_{args.strategy}_CV.json' if args.cross_validation else f'{args.output}_{args.strategy}.json' 
        with open(output_path, 'w') as f:
            json.dump(log, f, indent=4)
