import json, os, ast
from difflib import SequenceMatcher
from AutoFL import name_utils
from tqdm import tqdm
import torch
import argparse
from collections import defaultdict
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx

D4J_BUG_INFO_DIR = '../autofl/data/defects4j'
d4j_bugs = os.listdir('../autofl/data/defects4j')

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
            if len(function_calls) >= k:
                break

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
        
        # last_response = dialog[-1]
        # if last_response.get("role") and last_response.get("content") and not last_response.get("function_call"):
        raw_answer = dialog[-1]["content"]
        if raw_answer:
            answer_signatures_dict = dp.process_answer(raw_answer)
            for answer, signatures in answer_signatures_dict.items():
                for sig in signatures:
                    if sig:
                        arg_set.add(sig)
            reasoning_paths.append({"function_calls": function_calls, "answer": answer_signatures_dict})
        else: # raw_answer = one
            reasoning_paths.append({"function_calls": function_calls, "answer": raw_answer})

    return reasoning_paths, arg_set

def generate_LIG(reasoning_paths_dict, labels_dict, args_dict, k, arg_vector_size):
    def add_weighted_edge(G, u, v, weight = 1):
        if G.has_edge(u, v):
            G[u][v]['weight'] += weight
        else:
            G.add_edge(u, v, weight = weight)
        
    dataset_S = []
    dataset_F = []
    dataset_FA = []

    # arg_vector_size = 0
    # for arg_set in args_dict.values():
    #     if len(list(arg_set)) > arg_vector_size:
    #         arg_vector_size = len(list(arg_set))
    # arg_vector_size += 1 # for None

    for bug_name in reasoning_paths_dict.keys():
        # print(bug_name)
        reasoning_paths = reasoning_paths_dict[bug_name]
        arg_list = list(args_dict[bug_name])
        

        LIG = nx.DiGraph()
        for _, rp in enumerate(reasoning_paths):
            function_calls, answer = rp["function_calls"], rp["answer"]
            if len(function_calls) == 0:
                continue
            if not LIG.has_node(str(function_calls[0])):
                LIG.add_node(str(function_calls[0]))
            for i, fc in enumerate(function_calls[1:]):
                if fc == None:
                    print(rp)
                    print(bug_name)
                if i + 1 < k:
                    if not LIG.has_node(str(fc)):
                        LIG.add_node(str(fc))
                    add_weighted_edge(LIG, str(function_calls[i]), str(fc))
            
            if len(function_calls) < k:
                if answer:
                    for answers in answer.values():
                        for a in answers:
                            if not LIG.has_node(str(a)):
                                LIG.add_node(str(a))
                            add_weighted_edge(LIG, str(function_calls[-1]), str(a))
                else: # answer = None
                    if not LIG.has_node(str(answer)):
                        LIG.add_node(str(answer))
                    add_weighted_edge(LIG, str(function_calls[-1]), str(answer))
                    


        # save_LIG_image(LIG, filename=f"LIG_{k}_{bug_name}.png")

        S_data = from_networkx(LIG)
        F_data = from_networkx(LIG)
        FA_data = from_networkx(LIG)

        S_data.edge_attr = torch.tensor([LIG[u][v]['weight'] for u, v in LIG.edges()], dtype = torch.float)
        F_data.edge_attr = torch.tensor([LIG[u][v]['weight'] for u, v in LIG.edges()], dtype = torch.float)
        FA_data.edge_attr = torch.tensor([LIG[u][v]['weight'] for u, v in LIG.edges()], dtype = torch.float)

        S_nodes_x = []
        F_nodes_x = []
        FA_nodes_x = []

        for node_str in LIG.nodes():
                # print(node_str)
                # print(LIG.out_degree(node_str))
            try:
                node = ast.literal_eval(node_str)
            except:
                pass
            #     print(node_str)
            #     print(bug_name)
            #     # print('--')

            # Function call node
            if LIG.out_degree(node_str) == 0 and not isinstance(node, dict):
                func_vector = torch.ones(4, dtype=torch.float)
                
                answer_vector = torch.zeros(arg_vector_size, dtype=torch.float)
                if node != None:
                    answer_index = arg_list.index(node)
                    answer_vector[answer_index] = 1
                    # print("answer is None, but it should not")
                else:
                    # print("answer is None")
                    answer_vector[-1] = 1

                func_answer_vector = torch.cat((func_vector, answer_vector))

                F_nodes_x.append(func_vector)
                FA_nodes_x.append(func_answer_vector)
            else:
                # node = ast.literal_eval(node)
                # print("==")
                # print(node)
                # if not node:
                    # print(LIG.nodes())
                    # print(LIG.edges())
                    # print(bug_name)

                if node["name"] == "get_failing_tests_covered_classes":
                    func_vector = torch.tensor([1, 0, 0, 0], dtype=torch.float)
                elif node["name"] == "get_failing_tests_covered_methods_for_class":
                    func_vector = torch.tensor([0, 1, 0, 0], dtype=torch.float)
                elif node["name"] == "get_code_snippet":
                    func_vector = torch.tensor([0, 0, 1, 0], dtype=torch.float)
                elif node["name"] == "get_comments":
                    func_vector = torch.tensor([0, 0, 0, 1], dtype=torch.float)
            
                arg = node["arguments"]

                arg_vector = torch.zeros(arg_vector_size, dtype=torch.float)

                if arg:
                    arg_index = arg_list.index(arg)
                    arg_vector[arg_index] = 1
                elif node["name"] == "get_failing_tests_covered_classes":
                    pass
                else:
                    # print("argument is None")
                    arg_vector[-1] = 1
                func_arg_vector = torch.cat((func_vector, arg_vector))


                F_nodes_x.append(func_vector)
                FA_nodes_x.append(func_arg_vector)

            # Answer node
            # else:
                # func_vector = torch.ones(4, dtype=torch.float)
                
                # answer_vector = torch.zeros(arg_vector_size, dtype=torch.float)
                # if node != None:
                #     answer_index = arg_list.index(node)
                #     answer_vector[answer_index] = 1
                #     print("answer is None, but it should not")
                # else:
                #     print("answer is None")
                #     answer_vector[-1] = 1

                # func_answer_vector = torch.cat((func_vector, answer_vector))

                # F_nodes_x.append(func_vector)
                # FA_nodes_x.append(func_answer_vector)
            
            landscape_vector = torch.ones(4, dtype=torch.float)
            S_nodes_x.append(landscape_vector)

        S_x_stack = np.vstack(S_nodes_x)
        F_x_stack = np.vstack(F_nodes_x)
        FA_x_stack = np.vstack(FA_nodes_x)

        # print(torch.tensor(S_x_stack, dtype=torch.float).shape)
        S_data.x = torch.tensor(S_x_stack, dtype=torch.float)
        F_data.x = torch.tensor(F_x_stack, dtype=torch.float)
        FA_data.x = torch.tensor(FA_x_stack, dtype=torch.float)

        S_data.y = torch.tensor([labels_dict[bug_name]], dtype=torch.float)
        F_data.y = torch.tensor([labels_dict[bug_name]], dtype=torch.float)
        FA_data.y = torch.tensor([labels_dict[bug_name]], dtype=torch.float)
        # print(S_data.x.shape)
        # print(F_data.x.shape)
        # print(FA_data.x.shape)

        dataset_S.append(S_data)
        dataset_F.append(F_data)
        dataset_FA.append(FA_data)
    return dataset_S, dataset_F, dataset_FA

def main(model, repetition, num_files):
    ks = range(1, 13)
    # ks = [3]
    # ks = range(2, 13)
    all_gcn_S = dict()
    all_gcn_F = dict()
    all_gcn_FA = dict()
    arg_vector_size_dict = dict()
    for k in ks:
        all_gcn_S[k] = []
        all_gcn_F[k] = []
        all_gcn_FA[k] = []
        arg_vector_size_dict[k] = []

    for i in range(1, num_files+1):
        reasoning_paths_dict = dict()
        labels_dict = dict()
        args_dict = dict()
        for k in ks:
            reasoning_paths_dict[k] = dict()
            labels_dict[k] = dict()
            args_dict[k] = dict()

        if model == 'equal_weight':
            combined_result_file = f"../autofl/weighted_fl_results/accat1_de/equal_R{repetition}_{i}.json"
        else:
            combined_result_file = f"../autofl/weighted_fl_results/{model}/equal_R{repetition}_{i}.json"

        with open(combined_result_file, 'r') as f:
            combined_results = json.load(f)
        buggy_method_ranks = combined_results["ranks"]
        bug_list = list(buggy_method_ranks.keys())
        # bug_list = ['Chart_8']
        for bug_name in tqdm(bug_list):
            if buggy_method_ranks[bug_name] == 1:
                labels_dict[bug_name] = 1
            else:
                labels_dict[bug_name] = 0
            
            result_dirs = combined_results["sampled_dirs"]

            for k in ks:
                reasoning_paths, arg_set = d4j_get_reasoning_paths_and_args(result_dirs, bug_name, k)
                reasoning_paths_dict[k][bug_name] = reasoning_paths
                args_dict[k][bug_name] = arg_set
                arg_vector_size_dict[k].append(len(list(arg_set)))
        
        for k in ks:
            arg_vector_size = max(arg_vector_size_dict[k]) + 1
            gcn_S, gcn_F, gcn_FA = generate_LIG(reasoning_paths_dict[k], labels_dict, args_dict[k], k, arg_vector_size)
            all_gcn_S[k].extend(gcn_S)
            all_gcn_F[k].extend(gcn_F)
            all_gcn_FA[k].extend(gcn_FA)
            # print(len(all_gcn_S[k]))

        
    for k in ks:
        if not os.path.exists(f"./data/{model}/R{repetition}_{num_files}files/{k}"):
            os.makedirs(f"./data/{model}/R{repetition}_{num_files}files/{k}")

        torch.save({
            "dataset_S": all_gcn_S[k],
            "dataset_F": all_gcn_F[k],
            "dataset_FA": all_gcn_FA[k],
        }, f"data/{model}/R{repetition}_{num_files}files/{k}/gcn_dataset.pth")
        print(f"{k}th GCN datasets saved to gcn_dataset.pth")


                

        # print(max_args)
        # print(max_length)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='llama3')
    parser.add_argument('-r', '--repetition', default=1, type=int)
    parser.add_argument('-n', '--num_files', default=1, type=int)
    args = parser.parse_args()
    assert args.model in ['llama3', 'llama3.1', 'mistral-nemo', 'qwen2.5-coder', 'equal_weight']
    assert args.repetition in range(1, 25)
    assert args.num_files in range(1, 21)

    main(args.model, args.repetition, args.num_files)
