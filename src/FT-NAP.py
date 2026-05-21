import glob
import os

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.classes.mygraph import my_graph

# Bundled DAG corpora, anchored to the repository root. See the README for how
# to extract the .dot files from data/*.zip.
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

raw_ngrams_UD = []

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


for graphpath in tqdm(glob.glob(f'{_DATA_DIR}/under_development_dags/*.dot'), desc="Loading Graphs"):
    swf_name = graphpath.split('/')[-1].replace('.dot', '')
    try:
        query_graph = my_graph(graphpath)
    except Exception as e:
        print(f'Skipping unreadable graph {graphpath}: {e}')
        continue
    if len(query_graph.nodes) < 3:
        continue

    query_ngrams = query_graph.get_all_paths_with_edges_for_given_nodes(min_length=6,
                                                                        max_length=10,
                                                                        node_list=query_graph.find_roots())
    temp = []
    for ngram in query_ngrams:
        if len(ngram) <= 1 or ngram[-1] in nf_ops_list:
            continue
        filtered_ngram = [x for x in ngram if x.isupper()]
        if len(filtered_ngram) > 2: # Ensure there's at least a trace and a label after filtering
            temp.append(filtered_ngram)

    raw_ngrams_UD.extend(temp)



raw_ngrams_RELEASED = []

for graphpath in tqdm(glob.glob(f'{_DATA_DIR}/released/*.dot'), desc="Loading Graphs"):
    swf_name = graphpath.split('/')[-1].replace('.dot', '')
    try:
        query_graph = my_graph(graphpath)
    except Exception as e:
        print(f'Skipping unreadable graph {graphpath}: {e}')
        continue
    if len(query_graph.nodes) < 3:
        continue

    query_ngrams = query_graph.get_all_paths_with_edges_for_given_nodes(min_length=6,
                                                                        max_length=10,
                                                                        node_list=query_graph.find_roots())
    temp = []
    for ngram in query_ngrams:
        if len(ngram) <= 1 or ngram[-1] in nf_ops_list:
            continue
        filtered_ngram = [x for x in ngram if x.isupper()]
        if len(filtered_ngram) > 2: # Ensure there's at least a trace and a label after filtering
            temp.append(filtered_ngram)

    raw_ngrams_RELEASED.extend(temp)



MODEL_NAME = 'meta-llama/Llama-3.1-8B-Instruct'
# MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'
NUM_EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128

# Prepare Dataset for S-NAP
data_UD = []
data_RELEASED = []
for item_list in raw_ngrams_UD:
    # Ensure item_list has at least two elements (trace and label)
    if len(item_list) < 2:
        print(f"Skipping malformed data entry: {item_list} (requires at least a trace and a label)")
        continue
    trace = ", ".join(item_list[:-1])
    label = item_list[-1]
    data_UD.append({"trace": trace, "label": label})
for item_list in raw_ngrams_RELEASED:
    # Ensure item_list has at least two elements (trace and label)
    if len(item_list) < 2:
        print(f"Skipping malformed data entry: {item_list} (requires at least a trace and a label)")
        continue
    trace = ", ".join(item_list[:-1])
    label = item_list[-1]
    data_RELEASED.append({"trace": trace, "label": label})

train_traces = [d["trace"] for d in data_RELEASED]
train_labels = [d["label"] for d in data_RELEASED]
val_traces = [d["trace"] for d in data_UD]
val_labels = [d["label"] for d in data_UD]

all_labels = train_labels + val_labels

label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
encoded_train_labels = label_encoder.transform(train_labels)
encoded_val_labels = label_encoder.transform(val_labels)
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_


print(f"Number of classes for S-NAP TRAIN: {num_classes}")
print(f"Training samples: {len(train_traces)}")
print(f"Validation samples: {len(val_traces)}")


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "right"


class SNAPPredictionDataset(Dataset):
    def __init__(self, traces, labels, tokenizer, max_seq_length):
        self.traces = traces
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        trace = self.traces[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            trace,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = SNAPPredictionDataset(train_traces, encoded_train_labels, tokenizer, MAX_SEQ_LENGTH)
val_dataset = SNAPPredictionDataset(val_traces, encoded_val_labels, tokenizer, MAX_SEQ_LENGTH)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)

if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon (MPS) GPU for training.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA (CUDA) GPU for training.")
else:
    device = torch.device("cpu")
    print("Using CPU for training.")

model.to(device)

print(f"Model loaded and moved to: {device}")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

print("Starting fine-tuning for S-NAP classification...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

    print("\nStarting detailed validation...")

    model.eval()

    correct_predictions = 0
    wrong_predictions = 0
    unseen_class_errors = 0

    with torch.no_grad():
        for idx in tqdm(range(len(val_traces)), desc="Evaluating validation samples"):
            trace = val_traces[idx]
            true_label = val_labels[idx]

            try:
                encoding = tokenizer.encode_plus(
                    trace,
                    add_special_tokens=True,
                    max_length=MAX_SEQ_LENGTH,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_class_id = torch.argmax(logits, dim=-1).item()
                predicted_class_name = label_encoder.inverse_transform([predicted_class_id])[0]

                if true_label not in label_encoder.classes_:
                    unseen_class_errors += 1
                    continue

                if predicted_class_name == true_label:
                    correct_predictions += 1
                else:
                    wrong_predictions += 1

            except Exception as e:
                print(f"Error processing validation sample {idx}: {e}")
                unseen_class_errors += 1

    total = correct_predictions + wrong_predictions + unseen_class_errors
    accuracy = correct_predictions / total if total > 0 else 0.0

    print("\nValidation Results Summary:")
    print(f"Correct predictions     : {correct_predictions}")
    print(f"Wrong predictions       : {wrong_predictions}")
    print(f"Unseen class errors     : {unseen_class_errors}")
    print(f"Total validation samples: {total}")
    print(f"Accuracy                : {accuracy:.4f}")
