import json, os, ast
from difflib import SequenceMatcher
from AutoFL import name_utils
from tqdm import tqdm
import torch
import argparse
from collections import defaultdict

D4J_BUG_INFO_DIR = '../autofl/data/defects4j'

class D4JProcessing():
    def __init__(self, bug_name) -> None:
        self._method_lists = self._load_method_lists(bug_name)
        self._test_lists = self._load_test_lists(bug_name)
        self._field_lists = self._load_field_lists(bug_name)
        self._test_signatures = [test['signature'] for test in self._test_lists]
        self._field_signatures = [field['signature'] for field in self._field_lists] 
        self._method_signatures = [method['signature'] for method in self._method_lists]
    
    def process_get_failing_tests_covered_methods_for_class(self, class_name):

        for method in self._method_lists:
            if method["class_name"] == class_name:
                return class_name
            elif class_name in self._test_signatures:
                return class_name
            else:
                return None

    def process_get_code_snippet(self,signature):
        if signature in self._field_signatures:
            return signature

        method, candidates = self.get_matching_method_or_candidates(signature, 5)
        if method:
            return method['signature']
        
        if len(candidates) == 0 and not name_utils.is_method_signature(signature):
            candidates = [field for field in self._field_lists if name_utils.get_base_name(signature) in field["signature"]][:5]

        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return candidates[0]['signature']
        else:
            return None

    def process_get_comments(self, signature):
        if signature in self._field_signatures:
            return signature

        method, candidates = self.get_matching_method_or_candidates(signature, 5)
        if method:
            return method['signature']

        if len(candidates) == 0 and not name_utils.is_method_signature(signature):
            candidates = [field for field in self._field_lists if name_utils.get_base_name(signature) in field["signature"]][:5]

        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return candidates[0]['signature']
        else:
            return None

    def process_answer(self, answer):
        pred_exprs = answer.splitlines()
        matching_methods_signatures = {
            pred_expr: self.get_matching_method_signatures(pred_expr)
            for pred_expr in pred_exprs
        }

        return matching_methods_signatures

    def get_matching_method_or_candidates(self, pred_expr: str, num_max_candidates:int=None) -> tuple:
        candidates = {}

        short_method_name = name_utils.get_method_name(pred_expr)

        search_lists = []
        search_lists += self._method_lists
        search_lists += self._test_lists

        for method in search_lists:
            if name_utils.lenient_matcher(pred_expr, method['signature']):
                return (method, None)
            if short_method_name in method["signature"]:
                candidates[method["signature"]] = method

        if len(candidates) == 0:
            return None, []

        priority, candidate_signatures = self.get_highest_priority_candidates(
             pred_expr, list(candidates.keys()), num_max_candidates=num_max_candidates)

        assert (num_max_candidates is None or
                len(candidate_signatures) <= num_max_candidates)

        if priority == 0 and len(candidate_signatures) == 1:
            return (candidates[candidate_signatures[0]], None)
        else:
            return (None, [candidates[sig] for sig in candidate_signatures])

    def get_matching_method_signatures(self, pred_expr):
        return [
            signature for signature in self._method_signatures
            if name_utils.lenient_matcher(pred_expr, signature)
        ]
    
    def get_highest_priority_candidates(self, pred_expr: str, candidates: list,
                                        num_max_candidates:int=None):
        def _compute_similarity(method_name_1, arg_types_1, method_name_2, arg_types_2):
            # (method name similarity , short name matching, arg type similarity)
            return (
                SequenceMatcher(None, method_name_1, method_name_2).ratio(),
                method_name_1[-1] == method_name_2[-1],
                SequenceMatcher(None, arg_types_1, arg_types_2).ratio()
            )

        def _get_priority(method_similarity: float, short_name_match: bool,
                          arg_type_similarity: float):
            if method_similarity == 1.0:
                assert short_name_match
                priority = 0 if arg_type_similarity == 1.0 else 1
            else:
                priority = 2 if short_name_match else 3
            return priority

        assert len(candidates) > 0

        pred_method_name, pred_arg_types = name_utils.get_method_name_and_argument_types(pred_expr)
        similarities = defaultdict(list)
        for candidate in candidates:
            cand_method_name, cand_arg_types = name_utils.get_method_name_and_argument_types(candidate)
            similarity = _compute_similarity(pred_method_name, pred_arg_types,
                                            cand_method_name, cand_arg_types)
            priority = _get_priority(*similarity)
            similarities[priority].append((similarity, candidate))
        assert sum(len(v) for v in similarities.values()) == len(candidates)
        assert len(similarities) > 0

        highest_priority = min(similarities.keys())
        candidates = list(map(lambda t: t[1],
                              sorted(similarities[highest_priority], key=lambda t: t[0], reverse=True)))
        if num_max_candidates is not None:
            candidates = candidates[:num_max_candidates]
        return highest_priority, candidates
    
    def _load_method_lists(self, bug_name):
        with open(os.path.join(D4J_BUG_INFO_DIR, bug_name, "snippet.json")) as f:
            method_list = json.load(f)
        return method_list
    
    def _load_test_lists(self, bug_name):
        with open(os.path.join(D4J_BUG_INFO_DIR, bug_name, "test_snippet.json")) as f:
            test_list = json.load(f)
        return test_list
    
    def _load_field_lists(self, bug_name):
        with open(os.path.join(D4J_BUG_INFO_DIR, bug_name, "field_snippet.json")) as f:
            field_list = json.load(f)
        return field_list
    
def d4j_get_reasoning_paths_and_args(result_dirs, bug_name, k):
    arg_set = set()
    reasoning_paths = []
    dp = D4JProcessing(bug_name)

    for rd in result_dirs:
        result_file = os.path.join("../autofl", rd, f"XFL-{bug_name}.json")
        with open(result_file, 'r') as f:
            content = json.load(f)
            
        function_calls = []
        dialog = content["messages"]
        for j, m in enumerate(dialog):
            # if len(function_calls) >= k:
            #     break

            if m.get("function_call"):
                function_name = m["function_call"]["name"]
                function_args = json.loads(m["function_call"]["arguments"])
                

                if function_name == "get_failing_tests_covered_classes":
                    # print(**function_args)
                    reformated_arg = None
                elif function_name == "get_failing_tests_covered_methods_for_class":
                    reformated_arg = dp.process_get_failing_tests_covered_methods_for_class(**function_args)
                elif function_name == "get_code_snippet":
                    reformated_arg = dp.process_get_code_snippet(**function_args)
                elif function_name == "get_comments":
                    reformated_arg = dp.process_get_comments(**function_args)

                if reformated_arg:
                    arg_set.add(reformated_arg)
                processed_function_call = {"name": function_name, "arguments": reformated_arg}
                function_calls.append(processed_function_call)
        raw_answer = dialog[-1]["content"]
        if len(function_calls) > 11:
            print(rd, bug_name)
        try:
            answer_signatures_dict = dp.process_answer(raw_answer)
            for answer, signatures in answer_signatures_dict.items():
                for sig in signatures:
                    arg_set.add(sig)
            reasoning_paths.append({"function_calls": function_calls, "answer": answer_signatures_dict})
        except:
            reasoning_paths.append({"function_calls": function_calls, "answer": raw_answer})
            print(raw_answer)

    return reasoning_paths, arg_set

def main(model, repetition, num_files):
    reasoning_paths_dict = dict()
    labels_dict = dict()
    args_dict = dict()
    ks = range(1, 12)

    max_arg_len = 0
    root_dir = "../autofl/weighted_fl_results"

    all_files = []
    max_args = 0
    max_length = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath == root_dir:
            continue
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            all_files.append(full_path)
    for combined_result_file in all_files:
    # for i in range(1, num_files+1):
    # for 
    #     if model == 'equal_weight':
    #         combined_result_file = f"../autofl/weighted_fl_results/accat1_de/equal_R{repetition}_{i}.json"
    #     else:
    #         combined_result_file = f"../autofl/weighted_fl_results/{model}/equal_R{repetition}_{i}.json"

        with open(combined_result_file, 'r') as f:
            combined_results = json.load(f)
        
        try:
            buggy_method_ranks = combined_results["ranks"]
            bug_list = list(buggy_method_ranks.keys())
        except:
            bug_list = list(combined_results["validation_ranks"].keys())


        for bug in bug_list:
        #     if buggy_method_ranks[bug] == 1:
        #         labels_dict[bug] = 1
        #     else:
        #         labels_dict[bug] = 0

            try:
                result_dirs = combined_results["sampled_dirs"]
            except:
                print(combined_result_file)
            reasoning_paths, arg_set = d4j_get_reasoning_paths_and_args(result_dirs, bug, 1)

            for rp in reasoning_paths:
                if len(rp["function_calls"]) >= max_length:
                    max_length = len(rp["function_calls"])
                    print(f"max_length: {max_length}")
            if len(arg_set) >= max_args:
                max_args = len(arg_set)
                print(f"max_args: {max_args}")

        print(max_args)
        print(max_length)












if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='llama3')
    parser.add_argument('-r', '--repetition', default=1)
    parser.add_argument('-n', '--num_files', default=1)
    args = parser.parse_args()
    assert args.model in ['llama3', 'llama3.1', 'mistral-nemo', 'qwen2.5-coder', 'equal_weight']
    assert args.repetition in range(1, 25)
    assert args.num_files in range(1, 21)

    main(args.model, args.repetition, args.num_files)
