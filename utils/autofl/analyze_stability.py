import random
import argparse
import pandas as pd

from compute_score import *

def get_samples(result_dirs, run_count, sample_size):
    samples = []
    for _ in range(sample_size):
        while True:
            sample = sorted(random.sample(result_dirs, run_count))
            if sample in samples:
                continue
            samples.append(sample)
            break
        
    return samples

def add_sample_scores(samples, num_samples):
    summed_scores = dict()

    bugs = list(samples[0].keys())
    for bug in bugs:
        methods = list(samples[0][bug].keys())
        summed_scores[bug] = dict(zip(methods, [{'score': 0.0, 'aux_score': (0, 0)} for _ in range(len(methods))])) # Currently assuming that the auxiliary_score is computed by default
        for score in [sample[bug] for sample in samples]:
            for method in methods:
                summed_scores[bug][method]['score'] += score[method]['score'] / num_samples
                aux1, aux2 = summed_scores[bug][method]['aux_score']
                _, new_aux2 = score[method]['aux_score']
                summed_scores[bug][method]['aux_score'] = (aux1, aux2 + new_aux2) # aux1 stands for the # of failing tests
    
    return summed_scores

def get_result_for_a_sample(sampled_dirs, scores_for_each_run):
    label = '_'.join([dir.split('/')[1].split('_')[-1] for dir in sampled_dirs])

    method_scores = add_sample_scores([scores_for_each_run[dir] for dir in sampled_dirs], len(sampled_dirs))
    method_scores = assign_rank(method_scores)

    buggy_method_ranks = get_buggy_method_ranks(method_scores, key="autofl_rank")

    summary = {"label": label, "total": len(method_scores)}
    for n in range(1, 6):
        summary[f"acc@{n}"] = calculate_acc(buggy_method_ranks, key="autofl_rank", n=n)
    
    return summary

def caclculate_scores_for_each_run(result_dirs, language, verbose):
    individual_scores = dict()
    for dir in result_dirs:
        print(f'Now pre-processing {dir}...')
        json_files, autofl_scores = compute_autofl_scores([dir], verbose=verbose)

        if args.aux:
            method_scores = add_auxiliary_scores(json_files, autofl_scores, language,
                                                verbose=verbose)
        else:
            method_scores = add_auxiliary_scores(json_files, autofl_scores, language, 
                                                default_aux_score=0, verbose=verbose)
        individual_scores[dir] = method_scores
    print('Pre-processing is done!')
    return individual_scores
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('result_dirs', nargs="+", type=str)
    parser.add_argument('--run_count', '-R', type=int, default=5)
    parser.add_argument('--sample_size', '-N', type=int, default=10)
    parser.add_argument('--output', '-o', type=str, default="samples")
    parser.add_argument('--language', '-l', type=str, default="java")
    parser.add_argument('--verbose', '-v', action="store_true")
    parser.add_argument('--aux', '-a', action="store_true")
    args = parser.parse_args()
    assert args.language in ["java", "python"]
    
    scores_for_each_run = caclculate_scores_for_each_run(args.result_dirs, args.language, args.verbose)
    
    for run_count in range(1, args.run_count + 1):
        samples = get_samples(args.result_dirs, run_count, args.sample_size)
        
        results = []
        for sample in samples:
            results.append(get_result_for_a_sample(sample, scores_for_each_run))
        
        df = pd.DataFrame(results)
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
        df.to_csv(os.path.join(args.output, f'R{run_count}_N{args.sample_size}.csv'))