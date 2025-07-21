from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np
from tqdm import tqdm
import glob
from src.classes.mygraph import my_graph
import itertools
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score


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

    sentences = []
    for train_path in train_paths:
        sentences.append(' '.join(train_path))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    sequences = tokenizer.texts_to_sequences(sentences)

    input_sequences = []
    output_words = []
    for seq in sequences:
        for i in range(1, len(seq)):
            input_sequences.append(seq[:i])
            output_words.append(seq[i])

    maxlen = max(len(seq) for seq in input_sequences)
    x_train = pad_sequences(input_sequences, maxlen=maxlen)
    y_train = to_categorical(output_words, num_classes=vocab_size)

    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=20
    )

    # model.save(f'../../outputs/bidirectionalLSTM.keras')
    return model, test_paths

repeat = 10
precision_sum = 0
recall_sum = 0
acc_sum = 0
f1_sum = 0

SOURCE_CORPUS = under_development_source_path
print('Bidirectional LSTM')
print(SOURCE_CORPUS)

for i in tqdm(range(repeat)):
    model, test_paths = create_knowledgebase(SOURCE_CORPUS)
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

    test_sentences = []
    for query_ngram in test_paths:
        while query_ngram[-1] == '':
            query_ngram.pop()
            query_ngram.pop()
        if len(query_ngram) < 2:
            continue

        incomplete_ngram = query_ngram[:-1]
        missing_element = query_ngram[-1]

        sentence = ' '.join(query_ngram)
        test_sentences.append(sentence)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(test_sentences)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    sequences = tokenizer.texts_to_sequences(test_sentences)

    input_sequences = []
    output_words = []
    for seq in sequences:
        for i in range(1, len(seq)):
            input_sequences.append(seq[:i])
            output_words.append(seq[i])

    maxlen = max(len(seq) for seq in input_sequences)
    x_test = pad_sequences(input_sequences, maxlen=maxlen)
    y_test = to_categorical(output_words, num_classes=vocab_size)

    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=-1)
    y_test_labels = np.argmax(y_test, axis=-1)

    precision = precision_score(y_test_labels, y_pred,
                                average='micro')
    recall = recall_score(y_test_labels, y_pred, average='micro')
    acc = accuracy_score(y_test_labels, y_pred)
    f1 = f1_score(y_test_labels, y_pred, average='micro')

    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")

    index_to_word = {index: word for word, index in tokenizer.word_index.items()}

    precision_sum += precision
    recall_sum += recall
    acc_sum += acc
    f1_sum += f1
print(f'--------------------------------')
print(f'MY E2E Precision: ({precision_sum/repeat})')
print(f'MY E2E Recall: ({recall_sum/repeat})')
print(f'MY E2E Acc: ({acc_sum/repeat})')
print(f'MY E2E F1: ({f1_sum/repeat})')
