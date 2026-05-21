# FlowPilot: A Suggestion System for Designing Scientific Workflows

FlowPilot is a suggestion system that helps developers design Scientific
Workflows (SWFs) by recommending the next operator to implement. It complements
code-generating LLMs by enriching their context with relevant domain knowledge
retrieved from a corpus of historical workflows. FlowPilot indexes historical
workflows in a *Similarity Knowledge Base* (SKB), retrieves the sub-workflows
that match the current development context, and predicts the next operator with
a Markov-chain model. An online-learning loop updates the SKB from user
feedback.

This repository accompanies the paper:

> Mahdi Esmailoghli and Matthias Weidlich. 2026. **FlowPilot: A Suggestion
> System for Designing Scientific Workflows.** Proc. ACM Manag. Data 4, 1
> (SIGMOD), Article 39. https://doi.org/10.1145/3786653

## Abstract

Scientific Workflows (SWF) encapsulate data processing tasks by organizing
various tools, operators, and data in a logical flow. Due to their complex and
domain-specific nature, developing SWFs remains laborious. Current
code-generating large language models (Code LLMs) struggle to assist users in
developing these workflows. This limitation arises primarily from the
insufficient availability of relevant training data in public repositories,
making it challenging for LLMs to learn specialized patterns and domain-specific
logic. To address this, we propose FlowPilot, a suggestion system for developing
SWFs that assists developers by suggesting the next operator. Our system
complements Code LLMs by enriching the source code to generate more accurate
results. FlowPilot leverages a similarity knowledge base (SKB) that indexes
historical workflows to find the ones matching the current context. To generate
relevant recommendations, FlowPilot employs a statistical approach based on
Markov chains to identify the most likely next step. As a proof of concept, we
evaluated our system on NextFlow workflows and the results demonstrate the
effectiveness of FlowPilot by outperforming state-of-the-art code-generating
models, e.g., Llama-4, and traditional methods, e.g., association rule mining
techniques.

## Repository structure

```
FlowPilot/
├── data/                              # Workflow DAG corpora (.dot files)
│   ├── released/                      #   NF-Core released workflows        (54)
│   ├── under_development_dags/        #   NF-Core under-development workflows (25)
│   └── github_repos_except_nfcore/    #   GitHub NextFlow workflows (no nf-core) (873)
├── src/
│   ├── flowpilot.py                   # FlowPilot: the proposed suggestion system
│   ├── llm_baselines.py              # Retrieval-augmented Code-LLM baselines (via Ollama)
│   ├── fim.py                         # Frequent Item set Mining (FIM) baseline
│   ├── fim_l.py                       # FIM-L: association rules over KNN n-grams
│   ├── ccg.py                         # CCG distance-based retrieval baseline
│   ├── blstm.py                       # Bidirectional LSTM baseline
│   ├── FT-NAP.py                      # Fine-Tuned Next Activity Prediction baseline
│   ├── classes/
│   │   ├── mygraph.py                 # DAG parsing and n-gram (path) extraction
│   │   ├── hmm.py                     # Markov-chain model for next-operator prediction
│   │   ├── helper.py                  # Embedding model + caching utilities
│   │   ├── ollama.py                  # Thin wrapper around a local Ollama server
│   │   └── t5p.py                     # CodeT5+ model wrapper
│   └── Helper/
│       └── recommender_helper_functions.py  # Path extraction, embedding, indexing helpers
├── requirements.txt
└── LICENSE
```

## Installation

FlowPilot runs on **Python 3.10+** (the pinned dependency set was verified on
Python 3.12). A virtual environment is strongly recommended — installing into a
shared/base environment can break unrelated packages.

1. Install the system dependency **Graphviz** (required to build `pygraphviz`):

   ```bash
   # macOS
   brew install graphviz
   # Debian / Ubuntu
   sudo apt-get install graphviz graphviz-dev
   ```

2. Create an isolated environment and install the Python dependencies:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

   On macOS, if `pygraphviz` fails to find Graphviz, point it at the Homebrew
   prefix before installing:

   ```bash
   export CFLAGS="-I$(brew --prefix graphviz)/include"
   export LDFLAGS="-L$(brew --prefix graphviz)/lib"
   ```

   The core system needs only a subset of the requirements; the rest are used by
   individual baselines (see comments in `requirements.txt`). On first run, the
   `all-MiniLM-L6-v2` sentence-transformer model is downloaded automatically.

## Data

The `data/` directory ships the workflow corpora as Graphviz `.dot` files, where
each file is the data-flow DAG of one executable NextFlow workflow (operators are
node labels, intermediate data are edge labels). These correspond to the
**executable SWFs** in Table 2 of the paper:

| Corpus (directory)              | Paper corpus        | # DAGs |
|---------------------------------|---------------------|-------:|
| `released/`                     | NF-Core Released    |     54 |
| `under_development_dags/`       | NF-Core UD          |     25 |
| `github_repos_except_nfcore/`   | GitHub NF           |    873 |

**Metadata note.** The metadata-augmented features described in the paper (README
and `nextflow_schema.json` embeddings) rely on per-workflow metadata that is
**not** bundled here. FlowPilot therefore runs in **n-gram mode** by default,
which corresponds to the "N-Gram" configuration in Figure 3 (within ~4% of the
full system on average). The metadata features are enabled automatically if you
populate:

```
data/metadata/readme/{release,under_development,git}/<workflow>_README.md
data/metadata/schema/{release,under_development,git}/<workflow>_nextflow_schema.json
```

## Usage

Run the scripts as modules from the repository root (so the `src` package
resolves):

```bash
# FlowPilot: query the released corpus, build the SKB from the under-development
# corpus, and report accuracy/precision for K = 1..10.
python -m src.flowpilot --query released --index under_development

# Useful options:
#   --query {released,under_development,github}   corpus to draw queries from
#   --index {released,under_development,github}   corpus used to build the SKB
#   --empty-index                                 cold start + online learning
#   --cache                                       reuse cached artifacts (outputs/)
#   --alpha/--beta/--gamma                        Markov scoring weights (Sec. 5.4)
```

Baselines:

```bash
python -m src.fim       --query released                       # FIM            (Table 3)
python -m src.fim_l     --query released --index under_development   # FIM-L     (Table 4)
python -m src.ccg       --query under_development               # CCG-based      (Table 3)
python -m src.blstm     --query under_development               # bidir. LSTM    (Table 3/4)
python -m src.llm_baselines --query released --index under_development --llm-model llama3   # Code-LLM (Table 3/4)

# FT-NAP has a hyphen in its filename, so run it directly with the repo on PYTHONPATH:
PYTHONPATH=. python src/FT-NAP.py                              # FT-NAP          (Table 3)
```

Notes:
- **Code-LLM baselines** (`llm_baselines.py`) require a running
  [Ollama](https://ollama.com) server with the requested model pulled, e.g.
  `ollama pull llama3`. Select the model with `--llm-model`.
- **CodeT5+ / FT-NAP** download large HuggingFace models and benefit from a GPU.
- Intermediate artifacts (extracted paths, embeddings, the HNSW index) are cached
  under `outputs/` and reused when `--cache` is passed.

## Mapping to the paper

| Script                | Role in the paper                                            |
|-----------------------|-------------------------------------------------------------|
| `flowpilot.py`        | FlowPilot (Tables 3–5, Figures 3–4)                         |
| `llm_baselines.py`    | Retrieval-augmented Code-LLMs, few-shot (Tables 3, 4)       |
| `fim.py`              | Frequent Item set Mining, FIM (Table 3)                     |
| `fim_l.py`            | FIM-L, association rules over KNN n-grams (Table 4)         |
| `ccg.py`              | CCG distance-based retrieval (Table 3)                      |
| `blstm.py`            | Bidirectional LSTM (Tables 3, 4)                            |
| `FT-NAP.py`           | Fine-Tuned Next Activity Prediction (Table 3)              |
| `classes/t5p.py`      | CodeT5+ model wrapper                                        |

## Citation

```bibtex
@article{esmailoghli2026flowpilot,
  author  = {Esmailoghli, Mahdi and Weidlich, Matthias},
  title   = {{FlowPilot}: A Suggestion System for Designing Scientific Workflows},
  journal = {Proc. ACM Manag. Data},
  volume  = {4},
  number  = {1},
  series  = {SIGMOD},
  articleno = {39},
  year    = {2026},
  doi     = {10.1145/3786653}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
