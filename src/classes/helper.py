import os
import pickle

import torch
from sentence_transformers import SentenceTransformer

# Sentence embedding model used throughout FlowPilot (see paper, Sec. 4.4.1).
model = SentenceTransformer('all-MiniLM-L6-v2')

# Directory for cached intermediate artifacts (pickled paths/embeddings/index).
# Anchored to the repository root so it works regardless of the current
# working directory; created on demand.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(_REPO_ROOT, 'outputs', 'pickle_files')


def embed_path(path):
    """Embed an n-gram (sequence of operators and data elements) into a vector."""
    sentence = ' '.join(path)
    return torch.tensor(model.encode(sentence))


def embed_text(text):
    """Embed a free-text string (e.g. a README or schema) into a vector."""
    return torch.tensor(model.encode(str(text)))


def pickle_object(object_to_pickle, name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(os.path.join(CACHE_DIR, f'{name}.pickle'), 'wb') as handle:
            pickle.dump(object_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:  # caching is best-effort; never fail a run over it
        print(f'[warn] could not cache "{name}": {exc}')


def unpickle_object(name):
    with open(os.path.join(CACHE_DIR, f'{name}.pickle'), 'rb') as handle:
        return pickle.load(handle)
