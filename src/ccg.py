import glob
import os
from tqdm import tqdm
from src.classes.mygraph import my_graph
from src.classes.helper import *
import hnswlib
from src.classes.t5p import T5P
import itertools
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
from src.classes.hmm import HiddenMarkovModel
warnings.filterwarnings("ignore")
import numpy as np
from src.classes.code_aware_recommender import CodeAwareRecommender
from scipy.spatial.distance import cdist
from src.Helper.recommender_helper_functions import *
import matplotlib.pyplot as plt
import heapq

module_nf = {}
label_to_path = {}
label_to_swf = None
label_to_embedding = None
released_source_path = '../../datasets/dags/dot/released/'
under_development_source_path = '../../datasets/dags/dot/under_development_dags/'
git_source_path = '../../datasets/dags/dot/github_repos_except_nfcore/'
KB = None
max_path_length = 20
min_path_length = 4
swf_to_inout = None
swf_to_schema = None
embedding_size = 384

USE_NGRAM_EMBEDDING = True
USE_INOUT_EMBEDDING = True
USE_PARAMS_EMBEDDING = False
USE_README_EMBEDDING = True

embedding_config = str(USE_NGRAM_EMBEDDING) + str(USE_INOUT_EMBEDDING) + str(USE_PARAMS_EMBEDDING) + str(USE_README_EMBEDDING)

nf_ops_list = [x.upper() for x in list(
    {'branch', 'channel', 'collect', 'combine', 'emit', 'flatten', 'join', 'merge', 'output', 'scatter',
     'split', 'zip', 'map', 'filter', 'group', 'set', 'setval', 'mix', 'buffer', 'collate', 'collectFile',
     'concat', 'count', 'cross', 'distinct', 'emit', 'expand', 'filter', 'flatten', 'fold', 'group', 'head',
     'join', 'map', 'max', 'min', 'mix', 'output', 'pair', 'pick', 'reduce', 'reverse', 'sample', 'set',
     'setval', 'size', 'skip', 'sort', 'split', 'tail', 'take', 'toFile', 'toPath', 'toSet', 'toTuple',
     'unique', 'unzip', 'zip', 'countfasta', 'countFastq', 'countJson', 'countLines', 'cross', 'distinct',
     'dump', 'filter', 'first', 'flatmap', 'flatten', 'grouptuple', 'ifEmpty', 'join', 'last', 'merge', 'map',
     'max', 'min', 'mix', 'multiMap', 'randomSample', 'reduce', 'set', 'splitCsv', 'splitFasta', 'splitFastq',
     'splitJson', 'splitText', 'subscribe', 'sum', 'take', 'tap', 'toInteger', 'toList', 'toSortedList',
     'transpose', 'unique', 'until', 'view', ''})]

def parse_ngram(ngram):
    """
    Converts a raw list (with mixed nodes and data types) into a list of edges:
    (source_node, target_node, set_of_data_types)
    """
    result = []
    i = 0
    while i < len(ngram) - 1:
        src = ngram[i]
        next_item = ngram[i + 1]

        # If next item is a lowercase data type
        if next_item.islower():
            if i + 2 < len(ngram):
                tgt = ngram[i + 2]
                result.append((src, tgt, {next_item}))
                i += 3
            else:
                break  # malformed
        else:
            # It's a direct edge with no data type
            tgt = next_item
            result.append((src, tgt, set()))
            i += 2
    return result


def build_node_index(edges1, edges2):
    nodes = set()
    for e in edges1 + edges2:
        nodes.add(e[0])
        nodes.add(e[1])
    sorted_nodes = sorted(nodes)
    return {node: idx for idx, node in enumerate(sorted_nodes)}, len(sorted_nodes)


def build_matrix(edges, node_index, n, reference_edges):
    matrix = np.zeros((n, n))
    ref_map = {(e[0], e[1]): e[2] for e in reference_edges}

    for src, tgt, dtypes in edges:
        i, j = node_index[src], node_index[tgt]
        key = (src, tgt)

        if key in ref_map:
            if ref_map[key] == dtypes:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0.5  # same edge, different data types
        else:
            matrix[i][j] = 1  # unmatched edge
    return matrix

def workflow_similarity_ngram(raw_ngram1, raw_ngram2):
    edges1 = parse_ngram(raw_ngram1)
    edges2 = parse_ngram(raw_ngram2)

    node_index, n = build_node_index(edges1, edges2)

    M1 = build_matrix(edges1, node_index, n, edges2)
    M2 = build_matrix(edges2, node_index, n, edges1)

    # Frobenius norm
    diff = M1 - M2
    distance = np.linalg.norm(diff)

    # Normalize
    max_distance = np.linalg.norm(np.ones((n, n)))
    similarity = 1 - (distance / max_distance)
    return similarity

def query_KB(query_ngram, k):
    heap = []
    global KB

    for candidate_ngram in KB:
        score = workflow_similarity_ngram(query_ngram, candidate_ngram)

        if len(heap) < k:
            heapq.heappush(heap, (score, candidate_ngram))
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, candidate_ngram))

    top_k = sorted(heap, key=lambda x: x[0], reverse=True)
    return top_k

def add_new_datapoint_to_KB(ngram):
    global KB
    KB += [ngram]

def recommender(QUERY_CORPUS, INDEX_CORPUS, INDEX_CACHE, INDEX_EMPTY, leave_one_out = False, alpha=.7, beta=.01, gamma=.01):
    k_list = [10]
    top_n = 1

    ALLOW_LEARNING = True

    print(f'Source Corpus = {QUERY_CORPUS}')
    print('relaxed precision top n:', top_n)
    user_label_budget_per_swf = -1  # per SWF
    print('User label budget:', user_label_budget_per_swf)

    for k in k_list:
        global KB
        KB = []
        print(f'K = {k}')

        case_counter = 0
        match_counter = 0
        my_solution_counter = 0
        my_certain_solution_counter = 0
        my_non_certain_solution_counter = 0
        certain_case_counter = 0
        uncertainty_threshold = 0.5
        user_label_counter = 0
        list_of_correct_cases = []
        list_of_wrong_cases = []
        query_path_min_length = min_path_length
        query_path_max_length = max_path_length

        print('Uncertainty threshold:', uncertainty_threshold)

        sum_ratio_of_swf = 0  # SIGMA  # of swf/# of paths
        for graphpath in tqdm(glob.glob(f'{QUERY_CORPUS}*.dot')):
            swf_name = graphpath.split('/')[-1].replace('.dot', '')

            user_label_budget = user_label_budget_per_swf
            user_label_counter_per_swf = 0

            try:
                query_graph = my_graph(graphpath)
            except:
                print('Error in reading the graph')
                os.remove(graphpath)
                continue
            if len(query_graph.nodes) < 3:
                os.remove(graphpath)
                continue

            query_ngrams = query_graph.get_all_paths_with_edges_for_given_nodes(min_length=query_path_min_length + 2,
                                                                max_length=query_path_max_length, node_list=query_graph.find_roots())

            deduplicated_lists = []
            seen = set()
            for inner_list in query_ngrams:
                # Convert each inner list to a tuple so it can be added to a set
                tuple_rep = tuple(inner_list)
                if tuple_rep not in seen:
                    deduplicated_lists.append(inner_list)
                    seen.add(tuple_rep)
            query_ngrams = deduplicated_lists[:]

            query_ngrams = sorted(query_ngrams, key=lambda x: (x, len(x)))

            for ngram in query_ngrams[:1000]:
                only_ops_ngram = [x for x in ngram if x.upper() == x]
                if len(only_ops_ngram) < 3:
                    continue

                missing_element = only_ops_ngram[-1]
                missing_element_index = len(ngram) - 1 - ngram[::-1].index(missing_element)
                incomplete_ngram = ngram[:missing_element_index]

                if len(ngram) <= 1 or ngram[-1] in nf_ops_list:
                    continue

                knn_paths = []
                knn_distances = []
                results = query_KB(incomplete_ngram, k)
                for score, ngram in results:
                    knn_paths+=[ngram]
                    knn_distances+=[score]

                case_counter += 1

                match = False
                for path in knn_paths:
                    if missing_element in set(path):
                        match = True

                certain = False
                update_index = False
                if match:
                    match_counter += 1

                    my_hmm = HiddenMarkovModel(knn_paths, knn_distances, backwards_ratio=beta,
                                               distance_coefficient=gamma, path_distance_coefficient=alpha)
                    encoded_path, non_encoded_path = my_hmm.observations_to_stateid([incomplete_ngram])
                    if len(encoded_path) == 0:
                        continue

                    prob_list_of_current_state = my_hmm.transition_prob[encoded_path[-1]][0]
                    entropy = -np.sum(prob_list_of_current_state * np.log2(prob_list_of_current_state))
                    max_entropy = -np.sum(np.ones(len(prob_list_of_current_state)) * (
                                1 / len(prob_list_of_current_state) * np.log2(1 / len(prob_list_of_current_state))))
                    normalized_entropy = entropy / max_entropy
                    if max_entropy == 0:
                        normalized_entropy = 1

                    top_n_indexes = np.argsort(prob_list_of_current_state)[-top_n:][::-1]
                    markov_solution = [my_hmm.states[x] for x in top_n_indexes]

                    # Update the index if the uncertainty is high
                    if normalized_entropy > uncertainty_threshold and (
                            user_label_budget > user_label_counter_per_swf or user_label_budget <= 0):
                        update_index = True
                    else:
                        certain_case_counter += 1
                        certain = True

                    if missing_element in markov_solution:
                        list_of_correct_cases.append((case_counter, normalized_entropy))
                        my_solution_counter += 1
                        if certain:
                            my_certain_solution_counter += 1
                        else:
                            my_non_certain_solution_counter += 1

                    else:
                        list_of_wrong_cases.append((case_counter, normalized_entropy))

                        # Update the index if the prediction has been wrong
                        if user_label_budget > user_label_counter_per_swf or user_label_budget <= 0:
                            update_index = True
                else:
                    # Update the index if the next element doesn't exist in the current knn paths
                    if user_label_budget > user_label_counter_per_swf or user_label_budget <= 0:
                        update_index = True
                # update index
                if update_index & ALLOW_LEARNING:
                    user_label_counter += 1
                    user_label_counter_per_swf += 1
                    add_new_datapoint_to_KB(incomplete_ngram + [missing_element])

        print(f'--------------------------------')
        print(f'K = {k}')
        print(f'Total cases: {case_counter}')
        print(f'Match: {match_counter}')
        print(f'MY HMM Match: {my_solution_counter}')
        print('Certain cases:', certain_case_counter)
        print(f'Max knn recall: ({k}, {match_counter / case_counter})')
        print(f'MY HMM Precision: ({my_solution_counter / match_counter})')
        print(f'MY HMM Accuracy: ({k}, {(my_solution_counter / case_counter)})')
        print(f'MY HMM Precision among certain cases: {my_certain_solution_counter / certain_case_counter}')
        print(f'MY HMM Precision among NON-certain cases: {my_non_certain_solution_counter / (match_counter - certain_case_counter)}')
        print(f'Average ratio of number of swf to the number of paths: {sum_ratio_of_swf / case_counter}')
        print(f'User label required: {user_label_counter}')


def run():
    print('FlowPilot AL n-gram without data Under')
    recommender(QUERY_CORPUS=under_development_source_path, INDEX_CORPUS=under_development_source_path, INDEX_CACHE=False, INDEX_EMPTY=True, leave_one_out=False)

run()


