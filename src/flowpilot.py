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

# print(USE_NGRAM_EMBEDDING, USE_INOUT_EMBEDDING, USE_PARAMS_EMBEDDING, USE_README_EMBEDDING)

embedding_config = str(USE_NGRAM_EMBEDDING) + str(USE_INOUT_EMBEDDING) + str(USE_PARAMS_EMBEDDING) + str(USE_README_EMBEDDING)


def delete_items_from_KB(item_ids):
    for item_id in item_ids:
        try:
            global KB
            KB.mark_deleted(item_id)
        except:
            continue

def create_index(source, cache=False, empty_index = False, exclude_wf = ''):

    if exclude_wf != '':
        cache = False
        empty_index = False

    name = source.split('/')[-2]

    print('Create all possible paths for workflow graphs')
    if cache:
        swf_to_paths = unpickle_object(f'swf_to_paths_domain_min_length_{min_path_length}_max_length_{max_path_length}_{name}')
    else:
        swf_to_paths = create_paths(source, min_path_length, max_path_length, exclude_wf=exclude_wf)

        pickle_object(swf_to_paths, f'swf_to_paths_domain_min_length_{min_path_length}_max_length_{max_path_length}_{name}')


    print('Find the path embeddings for workflows')
    if cache:
        swf_path_embeddings = unpickle_object(f'swf_path_embeddings_domain_min_length_{min_path_length}_max_length_{max_path_length}_{name}')
    else:
        swf_path_embeddings = create_embeddings(swf_to_paths)
        pickle_object(swf_path_embeddings, f'swf_path_embeddings_domain_min_length_{min_path_length}_max_length_{max_path_length}_{name}')

    print('Embed README files per SWF')
    if cache:
        swf_to_readme_embedding = unpickle_object(f'swf_to_readme_embedding_{name}')
    else:
        if source == released_source_path:
            metadata_dir = '../../datasets/metadata/readme/release/'
        elif source == under_development_source_path:
            metadata_dir = '../../datasets/metadata/readme/under_development/'
        else:
            metadata_dir = '../../datasets/metadata/readme/git/'
        swf_to_readme_embedding = embed_readme_files(metadata_dir)
        pickle_object(swf_to_readme_embedding, f'swf_to_readme_embedding_{name}')

    print('Embed README files per SWF')
    global swf_to_inout
    if cache:
        swf_to_inout = unpickle_object(f'swf_to_inout_{name}')
    else:
        if source == released_source_path:
            metadata_dir = '../../datasets/metadata/schema/release/'
        elif source == under_development_source_path:
            metadata_dir = '../../datasets/metadata/schema/under_development/'
        else:
            metadata_dir = '../../datasets/metadata/schema/git/'
        swf_to_inout = get_swf_to_input_output(metadata_dir)
        pickle_object(swf_to_inout, f'swf_to_inout_{name}')

    print('Embed schema parameters per SWF')
    global swf_to_schema
    if cache:
        swf_to_schema = unpickle_object(f'swf_to_schema_{name}')
    else:
        if source == released_source_path:
            metadata_dir = '../../datasets/metadata/schema/release/'
        elif source == under_development_source_path:
            metadata_dir = '../../datasets/metadata/schema/under_development/'
        else:
            metadata_dir = '../../datasets/metadata/schema/git/'
        swf_to_schema = get_swf_to_schema(metadata_dir)
        pickle_object(swf_to_schema, f'swf_to_schema_{name}')

    print('index the representative vector of all paths')
    if cache:
        index = unpickle_object(f'hnsw_index_domain_min_length_{min_path_length}_max_length_{max_path_length}_{name}_{str(empty_index)}_{embedding_config}')
    embeddings_to_index = []
    for swf_path_embedding in swf_path_embeddings:
        inout_embedding = get_inout_embedding(swf_path_embedding[0])
        schema_embedding =  get_parameters_embedding(swf_path_embedding[0])
        readme_embedding = torch.zeros(len(schema_embedding))
        if swf_path_embedding[0] in swf_to_readme_embedding:
            readme_embedding = swf_to_readme_embedding[swf_path_embedding[0]]
        tuple_to_be_indexed = ()

        if USE_NGRAM_EMBEDDING:
            tuple_to_be_indexed = tuple_to_be_indexed + (swf_path_embedding[2],)

        tensor_list = []
        if USE_INOUT_EMBEDDING:
            tensor_list = tensor_list + [inout_embedding]
        if USE_PARAMS_EMBEDDING:
            tensor_list = tensor_list + [schema_embedding]
        if USE_README_EMBEDDING:
            tensor_list = tensor_list + [readme_embedding]

        if USE_INOUT_EMBEDDING or USE_PARAMS_EMBEDDING or USE_README_EMBEDDING:
            stacked_tensors = torch.stack(tensor_list)
            average_tensor = torch.mean(stacked_tensors, dim=0)
            tuple_to_be_indexed = tuple_to_be_indexed + (average_tensor,)
        embeddings_to_index.append(torch.cat(tuple_to_be_indexed, dim=0).numpy()) # ngram embedding, inout, parameters
    if not cache:
        index = index_embeddings(np.array(embeddings_to_index), range(len(embeddings_to_index)), empty_index)
        pickle_object(index, f'hnsw_index_domain_min_length_{min_path_length}_max_length_{max_path_length}_{name}_{str(empty_index)}_{embedding_config}')

    global KB
    KB = index

    global label_to_swf
    global label_to_path
    global label_to_embedding
    label_to_swf = dict(zip(range(len(swf_path_embeddings)), [x[0] for x in swf_path_embeddings]))
    label_to_path = dict(zip(range(len(swf_path_embeddings)), [x[1] for x in swf_path_embeddings]))
    label_to_embedding = dict(zip(range(len(embeddings_to_index)), embeddings_to_index))

def query_KB(ngram_embedding, swf_inout_embedding, swf_parameters_embedding, readme_embedding, k):
    tuple_to_be_queried = ()
    if USE_NGRAM_EMBEDDING:
        tuple_to_be_queried = tuple_to_be_queried + (ngram_embedding,)
    tensor_list = []
    if USE_INOUT_EMBEDDING:
        tensor_list = tensor_list + [swf_inout_embedding]
    if USE_PARAMS_EMBEDDING:
        tensor_list = tensor_list + [swf_parameters_embedding]
    if USE_README_EMBEDDING:
        tensor_list = tensor_list + [readme_embedding]
    # tensor_list = [swf_inout_embedding, swf_parameters_embedding, readme_embedding]
    if USE_INOUT_EMBEDDING or USE_PARAMS_EMBEDDING or USE_README_EMBEDDING:
        stacked_tensors = torch.stack(tensor_list)
        average_tensor = torch.mean(stacked_tensors, dim=0)
        tuple_to_be_queried = tuple_to_be_queried + (average_tensor,)

    full_embedding = torch.cat(tuple_to_be_queried, dim=0).numpy()
    # ngram_embedding = torch.cat((ngram_embedding, torch.zeros(len(swf_inout_embedding)), torch.zeros(len(swf_parameters_embedding))), dim=0).numpy()

    knn_labels, knn_distances = KB.knn_query(full_embedding, k=k, num_threads=12)
    knn_labels = knn_labels[0]
    knn_distances = knn_distances[0]

    return knn_labels, knn_distances

    #re-rank the topk based on their explanation and input and output
    knn_labels_to_domain_embedding = {}
    for label in knn_labels:
        knn_labels_to_domain_embedding[label] = np.array(label_to_embedding[label][:len(swf_inout_embedding)+len(swf_parameters_embedding)+len(readme_embedding)])
    sorted_list_label_embedding = sorted(knn_labels_to_domain_embedding.items())
    query_domain_embedding = np.hstack((swf_inout_embedding, swf_parameters_embedding))

    # Compute L2 distance (or cosine similarity) between query and retrieved candidates
    distances12 = cdist(np.array([query_domain_embedding]), np.array([x[1] for x in sorted_list_label_embedding]), metric='euclidean')[0]

    # Re-rank indices based on feature1 & feature2 distances
    sorted_indices = np.argsort(distances12)  # Sort by distance (lower is better)
    final_labels = knn_labels[sorted_indices]  # Reorder based on new ranking
    final_distances = knn_distances[sorted_indices]  # Reorder based on new ranking

    return final_labels[:k], final_distances[:k]

def get_inout_embedding(swf_name, size=embedding_size):
    if swf_name in swf_to_inout:
        return swf_to_inout[swf_name]
    return torch.zeros(size)

def get_parameters_embedding(swf_name, size=embedding_size):
    if swf_name in swf_to_schema:
        return swf_to_schema[swf_name]
    return torch.zeros(size)

def fetch_inout_embedding(query_path, swf_name, size=embedding_size):
    if query_path == released_source_path:
        file_path = f'../../datasets/metadata/schema/release/{swf_name}_nextflow_schema.json'
    elif query_path == under_development_source_path:
        file_path = f'../../datasets/metadata/schema/under_development/{swf_name}_nextflow_schema.json'
    else:
        file_path = f'../../datasets/metadata/schema/git/{swf_name}_nextflow_schema.json'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            json_content = json.load(file)
            if 'definitions' in json_content and 'input_output_options' in json_content['definitions']:
                return embed_text(str(json_content['definitions']['input_output_options']))
    return torch.zeros(size)

def fetch_parameters_embedding(query_path, swf_name, size=embedding_size):
    if query_path == released_source_path:
        file_path = f'../../datasets/metadata/schema/release/{swf_name}_nextflow_schema.json'
    elif query_path == under_development_source_path:
        file_path = f'../../datasets/metadata/schema/under_development/{swf_name}_nextflow_schema.json'
    else:
        file_path = f'../../datasets/metadata/schema/git/{swf_name}_nextflow_schema.json'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            json_content = json.load(file)
            return embed_text(str(json_content))
    return torch.zeros(size)

def fetch_readme_embedding(query_path, swf_name, size=embedding_size):
    if query_path == released_source_path:
        file_path = f'../../datasets/metadata/readme/release/{swf_name}_README.md'
    elif query_path == under_development_source_path:
        file_path = f'../../datasets/metadata/readme/under_development/{swf_name}_README.md'
    else:
        file_path = f'../../datasets/metadata/readme/git/{swf_name}_README.md'
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
            return embed_text(str(content))
    return torch.zeros(size)

def add_new_datapoint_to_KB(new_path, source_path, swf, size=embedding_size):
    new_embedding = embed_path(new_path)
    inout_embedding = fetch_inout_embedding(source_path, swf)
    schema_embedding =  torch.zeros(size)
    readme_embedding = torch.zeros(size)
    tuple_to_be_added = ()
    if USE_NGRAM_EMBEDDING:
        tuple_to_be_added = tuple_to_be_added + (new_embedding,)
    tensor_list = []
    if USE_INOUT_EMBEDDING:
        tensor_list = tensor_list + [inout_embedding]
    if USE_PARAMS_EMBEDDING:
        tensor_list = tensor_list + [schema_embedding]
    if USE_README_EMBEDDING:
        tensor_list = tensor_list + [readme_embedding]

    # tensor_list = [inout_embedding, schema_embedding, readme_embedding]
    if USE_INOUT_EMBEDDING or USE_PARAMS_EMBEDDING or USE_README_EMBEDDING:
        stacked_tensors = torch.stack(tensor_list)
        average_tensor = torch.mean(stacked_tensors, dim=0)
        tuple_to_be_added = tuple_to_be_added + (average_tensor,)

    final_embedding = torch.cat(tuple_to_be_added, dim=0).numpy()
    datapoint_id = len(label_to_path)
    label_to_swf[datapoint_id] = swf
    label_to_path[datapoint_id] = new_path
    label_to_embedding[datapoint_id] = final_embedding
    KB.add_items([final_embedding], [datapoint_id])

def recommender(QUERY_CORPUS, INDEX_CORPUS, INDEX_CACHE, INDEX_EMPTY, leave_one_out = False, alpha=.7, beta=.01, gamma=.01):
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    top_n = 1

    ALLOW_LEARNING = True
    ALLOW_DELETING = False

    SHOW_GRAPH = False

    print(f'Source Corpus = {QUERY_CORPUS}')
    print('relaxed precision top n:', top_n)
    user_label_budget_per_swf = -1  # per SWF
    print('User label budget:', user_label_budget_per_swf)

    for k in k_list:
        print(f'K = {k}')
        if not leave_one_out:
            create_index(INDEX_CORPUS, INDEX_CACHE, INDEX_EMPTY)

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


        max_number_of_predictions_per_swf = 100
        number_of_swfs = len(glob.glob(f'{QUERY_CORPUS}*.dot'))
        swf_counter = 0
        plot_results = np.zeros((max_number_of_predictions_per_swf, number_of_swfs, 2), dtype=int)

        print('Uncertainty threshold:', uncertainty_threshold)

        uncertainty_sum = 0.0

        sum_ratio_of_swf = 0  # SIGMA  # of swf/# of paths
        for graphpath in tqdm(glob.glob(f'{QUERY_CORPUS}*.dot')):
            swf_name = graphpath.split('/')[-1].replace('.dot', '')

            if leave_one_out:
                create_index(INDEX_CORPUS, INDEX_CACHE, INDEX_EMPTY, exclude_wf=swf_name)

            swf_prediction_counter = 0
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
            swf_inout_embedding = fetch_inout_embedding(QUERY_CORPUS, swf_name)
            swf_params_embedding = fetch_parameters_embedding(QUERY_CORPUS, swf_name)

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
                incomplete_path_embedding = embed_path(incomplete_ngram)

                if KB.get_current_count() < k:
                    knn_labels = []
                    knn_distances = []
                else:

                    description_embedding = fetch_readme_embedding(QUERY_CORPUS, swf_name, len(incomplete_path_embedding))
                    knn_labels, knn_distances = query_KB(incomplete_path_embedding, swf_inout_embedding, swf_params_embedding, description_embedding, k)
                knn_paths = []
                knn_swfs = []
                for label in knn_labels:
                    knn_paths.append(label_to_path[label])
                    knn_swfs.append(label_to_swf[label])

                case_counter += 1

                if len(knn_paths) > 0:
                    sum_ratio_of_swf += len(set(knn_swfs)) / len(knn_paths)

                correct_prediction_flag = False
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
                    uncertainty_sum += normalized_entropy

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
                        correct_prediction_flag = True
                        list_of_correct_cases.append((case_counter, normalized_entropy))
                        my_solution_counter += 1
                        if certain:
                            my_certain_solution_counter += 1
                        else:
                            my_non_certain_solution_counter += 1

                    else:
                        list_of_wrong_cases.append((case_counter, normalized_entropy))
                        # Update the index by deleting the current knn paths if the prediction is wrong and the uncertainty is low
                        if certain and ALLOW_DELETING and ALLOW_LEARNING:  # NEED MORE SOPHISTICATED WAY TO DELETE
                            ids_with_last_and_without_missing_elements = []
                            for knn_label_index in range(len(knn_labels)):
                                if missing_element not in label_to_path[knn_labels[knn_label_index]] and \
                                        non_encoded_path[-1] in label_to_path[knn_labels[knn_label_index]]:
                                    ids_with_last_and_without_missing_elements.append(knn_labels[knn_label_index])
                            delete_items_from_KB(ids_with_last_and_without_missing_elements)
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
                    add_new_datapoint_to_KB(incomplete_ngram + [missing_element], QUERY_CORPUS, swf_name)

                if swf_prediction_counter < max_number_of_predictions_per_swf:
                    plot_results[swf_prediction_counter][swf_counter][0] = int(match)
                    plot_results[swf_prediction_counter][swf_counter][1] = int(correct_prediction_flag)
                    swf_prediction_counter+=1

            swf_counter+=1
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
        print('KB size:', KB.get_current_count())
        print('Average uncertainty:', uncertainty_sum / match_counter)

        if SHOW_GRAPH:
            # Create the plot
            plt.figure(figsize=(10, 6))

            # Plot precision and recall lines
            matches_per_prediction_counter = np.array([np.sum(plot_results[x, :, 0] == 1) for x in np.arange(max_number_of_predictions_per_swf)])
            corrects_per_prediction_counter = np.array([np.sum(plot_results[x, :, 1] == 1) for x in np.arange(max_number_of_predictions_per_swf)])

            plt.plot(np.arange(max_number_of_predictions_per_swf), corrects_per_prediction_counter/matches_per_prediction_counter, marker='o', color='blue', label='Precision')
            plt.plot(np.arange(max_number_of_predictions_per_swf), matches_per_prediction_counter/number_of_swfs, marker='s', color='orange', label='Recall')
            plt.plot(np.arange(max_number_of_predictions_per_swf), corrects_per_prediction_counter/number_of_swfs, marker='*', color='olive', label='Acc')

            # Adding labels and title
            plt.title('Precision and Recall vs Counter')
            plt.xlabel('Counter')
            plt.ylabel('Value')

            # Show legend
            plt.legend()

            # Show grid for better readability
            plt.grid(True, linestyle='--', alpha=0.6)

            # Display the plot
            plt.show()


def run():
    print('FlowPilot AL n-gram without data Under')
    recommender(QUERY_CORPUS=released_source_path, INDEX_CORPUS=under_development_source_path, INDEX_CACHE=True, INDEX_EMPTY=True, leave_one_out=False)

run()


