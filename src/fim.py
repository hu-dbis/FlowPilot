import argparse
import glob
import itertools
import os

import pandas as pd
from tqdm import tqdm
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split

from src.classes.mygraph import my_graph

# Bundled DAG corpora, anchored to the repository root. See the README for how
# to extract the .dot files from data/*.zip.
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
released_source_path = f'{_DATA_DIR}/released/'
under_development_source_path = f'{_DATA_DIR}/under_development_dags/'
git_source_path = f'{_DATA_DIR}/github_repos_except_nfcore/'
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

def main():
    corpora = {
        'released': released_source_path,
        'under_development': under_development_source_path,
        'github': git_source_path,
    }
    parser = argparse.ArgumentParser(
        description='Run the Frequent Item set Mining (FIM) baseline (paper, Sec. 8.2).')
    parser.add_argument('--query', choices=corpora, default='released',
                        help='Corpus to mine association rules from.')
    parser.add_argument('--repeat', type=int, default=10,
                        help='Number of random train/test splits to average over.')
    parser.add_argument('--top-n', type=int, default=1,
                        help='Number of suggestions considered correct (recall@n).')
    args = parser.parse_args()

    source_corpus = corpora[args.query]
    repeat = args.repeat
    top_n = args.top_n
    precision_sum = 0
    print('AR Original')
    print(source_corpus)
    print(top_n)

    for i in tqdm(range(repeat)):
        KB, test_paths = create_knowledgebase(source_corpus)
        case_counter = 0
        my_solution_counter = 0
        nextflow_operators_case_counter = 0
        complex_operators_case_counter = 0
        nextflow_operators_correct_case_counter = 0
        complex_operators_correct_case_counter = 0

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
    print(f'MY E2E Recall: ({precision_sum / repeat})')


if __name__ == '__main__':
    main()
