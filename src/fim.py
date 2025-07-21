from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np
from tqdm import tqdm
import glob
from src.classes.mygraph import my_graph
import itertools
from sklearn.model_selection import train_test_split

released_source_path = '../../datasets/dags/dot/released/'
under_development_source_path = '../../datasets/dags/dot/under_development_dags/'
github_repos_source_path = '../../datasets/dags/dot/github_repos/'
github_repos_except_nfcore_source_path = '../../datasets/dags/dot/github_repos_except_nfcore/'
snakemake_source_path = '../../datasets/dags/dot/snakemake_dags/'
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


def create_knowledgebase(corpus_path):
    all_paths = []
    for graphpath in glob.glob(f'{corpus_path}*.dot'):
        try:
            graph = my_graph(graphpath)
        except:
            continue
        g_paths = graph.get_all_paths_with_edges(min_length=4, max_length=10)
        g_paths.sort()
        g_paths = list(k for k, _ in itertools.groupby(g_paths))
        all_paths.append(g_paths)
    all_paths = [x for sublist in all_paths for x in sublist]
    all_paths = [[item for item in path if item != '' and item.upper() == item] for path in all_paths]
    train_paths, test_paths = train_test_split(all_paths,test_size=0.2)

    a = TransactionEncoder()
    a_data = a.fit(train_paths).transform(train_paths)
    df = pd.DataFrame(a_data, columns=a.columns_)
    df = df.replace(False, 0)
    df = df.fillna(0)
    df = apriori(df, min_support=0.005, use_colnames=True, verbose=1)
    df_ar = association_rules(df, metric="confidence", min_threshold=0.5, num_itemsets=len(df))
    return df_ar, test_paths

repeat = 10
precision_sum = 0

top_n = 1
SOURCE_CORPUS = released_source_path
print('AR Original')
print(SOURCE_CORPUS)
print(top_n)

for i in tqdm(range(repeat)):
    KB, test_paths = create_knowledgebase(SOURCE_CORPUS)
    case_counter = 0
    my_solution_counter = 0
    my_certain_solution_counter = 0
    my_non_certain_solution_counter = 0
    nextflow_operators_case_counter = 0
    complex_operators_case_counter = 0
    nextflow_operators_correct_case_counter = 0
    complex_operators_correct_case_counter = 0
    certain_case_counter = 0
    uncertainty_threshold = 0.5
    user_label_counter = 0

    for query_ngram in test_paths:
        while query_ngram[-1] == '':
            query_ngram.pop()
            query_ngram.pop()
        if len(query_ngram) < 2:
            continue

        incomplete_ngram = query_ngram[:-1]
        missing_element = query_ngram[-1]

        ar_results = list(KB[KB['antecedents'] == set(incomplete_ngram)]['consequents'])
        ar_results = [item for sublist in ar_results for item in sublist]

        case_counter += 1


        if missing_element in nf_ops_list:
            nextflow_operators_case_counter += 1
        else:
            complex_operators_case_counter += 1

        if missing_element in ar_results[-top_n:]:
            my_solution_counter += 1

            if missing_element in nf_ops_list:
                nextflow_operators_correct_case_counter += 1
            else:
                complex_operators_correct_case_counter += 1
    precision_sum += my_solution_counter / case_counter
print(f'--------------------------------')
print(f'MY E2E Recall: ({precision_sum/repeat})')
