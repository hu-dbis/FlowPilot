from torchgen.operator_versions.gen_mobile_upgraders import sort_upgrader
from tqdm import tqdm
import glob
import itertools
from src.classes.mygraph import my_graph
from src.classes.helper import *
import hnswlib
import os
import json
from ollama import chat
from ollama import ChatResponse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def create_paths(corpus_path, min_path_length, max_path_length, exclude_wf = ''):
    '''
    :return: Nodesubset including the labels and the edges subset including the labels
    '''
    workflow_to_paths_dict = {}
    for graphpath in tqdm(glob.glob(f'{corpus_path}*.dot')):
        if exclude_wf != '' and exclude_wf in graphpath:
            continue
        try:
            graph = my_graph(graphpath)
        except:
            continue
        workflow_name = graphpath.split('/')[-1].replace('.dot', '')

        all_paths = graph.get_all_paths_with_edges(min_length=min_path_length, max_length=max_path_length)
        all_paths.sort()
        all_paths = list(k for k, _ in itertools.groupby(all_paths))
        workflow_to_paths_dict[workflow_name] = all_paths
        # graph.convert_edges_to_labeled_intermediate_nodes()
        # subgraph_nodes_with_labels_and_edges = graph.generate_connected_subgraphs()
        # workflow_to_subgraphs_dict[workflow_name] = subgraph_nodes_with_labels_and_edges # ([(nodeid, label), ...], [edges])
    return workflow_to_paths_dict

def create_paths_only_modules(corpus_path, min_path_length, max_path_length, exclude_wf = ''):
    '''
    :return: Nodesubset including the labels and the edges subset including the labels
    '''
    workflow_to_paths_dict = {}
    for graphpath in tqdm(glob.glob(f'{corpus_path}*.dot')):
        if exclude_wf != '' and exclude_wf in graphpath:
            continue
        try:
            graph = my_graph(graphpath)
        except:
            continue
        workflow_name = graphpath.split('/')[-1].replace('.dot', '')

        all_paths = graph.get_all_paths_with_edges(min_length=min_path_length, max_length=max_path_length)
        all_paths_only_modules = []
        for path in all_paths:
            temp_list = []
            for x in path:
                if x.upper() == x:
                    temp_list+=[x]
            all_paths_only_modules += [temp_list]
        all_paths_only_modules.sort()
        all_paths_only_modules = list(k for k, _ in itertools.groupby(all_paths_only_modules))
        workflow_to_paths_dict[workflow_name] = all_paths_only_modules
        # graph.convert_edges_to_labeled_intermediate_nodes()
        # subgraph_nodes_with_labels_and_edges = graph.generate_connected_subgraphs()
        # workflow_to_subgraphs_dict[workflow_name] = subgraph_nodes_with_labels_and_edges # ([(nodeid, label), ...], [edges])
    return workflow_to_paths_dict

def create_embeddings(workflow_to_paths_dict):
    '''
    :param workflow_to_paths_dict:
    :return: returns a list of triplets (wf_name, path, embedding)
    '''
    workflow_path_embeddings = []
    seen = set()

    for workflow_name in tqdm(workflow_to_paths_dict):
        for path in workflow_to_paths_dict[workflow_name]:
            if tuple(path) not in seen:
                workflow_path_embeddings.append((workflow_name, path, embed_path(path)))
                seen.add(tuple(path))
    return workflow_path_embeddings

def index_embeddings(embeddings, labels, empty_index=False):
    dimension = embeddings.shape[1]
    number_of_elements = len(embeddings)+1000000
    index_structure = hnswlib.Index(space='l2', dim=dimension)
    index_structure.init_index(max_elements=number_of_elements,
                               ef_construction=2000, M=160, allow_replace_deleted=False)
    index_structure.set_ef(200)  # ef should always be > k
    if not empty_index:
        index_structure.add_items(embeddings, ids=labels, num_threads=12)

    return index_structure

def embed_readme_files(source_path):
    swf_to_readme_embedding = {}
    for filename in os.listdir(source_path):
        if filename.endswith('.md'):
            with open(os.path.join(source_path, filename), 'r') as file:
                content = file.read()
                swf_to_readme_embedding[filename.replace('_README.md', '').split('-')[0]] = embed_text(content)
    return swf_to_readme_embedding

def get_swf_to_input_output(source_path):
    swf_to_inout = {}
    for filename in os.listdir(source_path):
        if filename.endswith('.json'):
            with open(os.path.join(source_path, filename), 'r') as file:
                json_content = json.load(file)
                if 'definitions' in json_content and 'input_output_options' in json_content['definitions']:
                    swf_to_inout[filename.replace('_nextflow_schema.json', '').split('-')[0]] = embed_text(str(json_content['definitions']['input_output_options']))
    return swf_to_inout

def get_swf_to_schema(source_path):
    swf_to_schema = {}
    for filename in os.listdir(source_path):
        if filename.endswith('.json'):
            with open(os.path.join(source_path, filename), 'r') as file:
                swf_to_schema[filename.replace('_nextflow_schema.json', '').split('-')[0]] = embed_text(file.read())
    return swf_to_schema

def generate_description(code):
    ollama_model = 'stable-code'
    prompt = f'explain this piece of code: {code}'
    response: ChatResponse = chat(model=ollama_model, messages=[
      {
        'role': 'user',
        'content': prompt,
      },
    ])
    return response.message.content

def generate_description_based_on_ngram(ngram):
    ollama_model = 'stable-code'
    prompt = f'I have a NextFlow workflow that contains these tasks and data: {" ".join(ngram)}. can you explain the functionality of the workflow?'
    response: ChatResponse = chat(model=ollama_model, messages=[
      {
        'role': 'user',
        'content': prompt,
      },
    ])
    return response.message.content

def create_tf_idf(workflow_to_paths_dict):
    # TOBE FIXED
    '''
        :param workflow_to_paths_dict:
        :return: returns a list of triplets (wf_name, path, vector)
        '''
    # Step 0: preprocessing of removing data info
    new_dict = {}
    for swf in workflow_to_paths_dict:
        new_dict[swf] = []
        for path in workflow_to_paths_dict[swf]:
            new_dict[swf] += [[x for x in path if x.upper() == x]]
    workflow_to_paths_dict = new_dict

    texts = []
    label_to_swf = {}
    label_to_path = {}
    idx = 0

    for swf_name, paths in workflow_to_paths_dict.items():
        for path in paths:
            text = " ".join(path)
            texts.append(text)
            label_to_swf[idx] = swf_name
            label_to_path[idx] = path
            idx += 1

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()

    return tfidf_matrix, label_to_swf, label_to_path, vectorizer


from sklearn.preprocessing import normalize


