# GNN-Based BERT for Understanding Context from Music

A supervised neural-network project implementing and comparing BERT, GraphSAGE, CNN, and multimodal fusion models for multi-label music-tag understanding.

## 1. Project overview

The project studies whether textual context and graph-based audio representations can be combined to improve music-tag prediction.

The completed experiment compares:

1. Majority baseline
2. CNN mel-spectrogram baseline
3. GraphSAGE (GNN)
4. Context BERT
5. Early-concatenation BERT + GNN fusion
6. Cross-attention BERT + GNN fusion

The final target vocabulary contains the **20 most frequent tags in the training split**.

The final leakage-free supervised dataset contains:

| Split | Samples |
|---|---:|
| Training | 393 |
| Validation | 44 |
| Test | 50 |
| Total | 487 |

The graph preprocessing pipeline produced **7,994 graph files**.

---

## 2. Final results

All primary F1 results use a fixed probability threshold of 0.5.

| Model | Macro-F1 | Micro-F1 | Macro AUC-PR | Micro AUC-PR |
|---|---:|---:|---:|---:|
| Majority | 0.0000 | 0.0000 | 0.0780 | 0.1465 |
| CNN | 0.0067 | 0.0217 | 0.1397 | 0.1594 |
| GNN | 0.0435 | 0.1373 | **0.2754** | 0.2244 |
| Context BERT | 0.0654 | **0.2626** | 0.2610 | 0.3763 |
| Early Fusion | 0.0489 | 0.1573 | 0.2536 | 0.3297 |
| **Cross-Attention Fusion** | **0.1128** | 0.2424 | **0.2820** | **0.4339** |

### Main finding

Cross-attention fusion gives the strongest:

- Macro-F1: **0.1128**
- Macro AUC-PR: **0.2820**
- Micro AUC-PR: **0.4339**

Context BERT gives the strongest Micro-F1:

- Micro-F1: **0.2626**

The result supports explicit interaction between textual and graph representations over simple embedding concatenation, although the small usable dataset means the results should be treated as exploratory.

---

## 3. Directory structure

```text
gnn_bert_music/
├── data/
│   ├── raw/
│   │   ├── fma_small/
│   │   └── fma_metadata/
│   ├── processed/
│   │   └── graphs/
│   └── splits/
│       ├── labels.json
│       ├── dataset.csv
│       └── context_dataset.csv
│
├── models/
│   ├── context_bert_model.pt
│   ├── gnn_model.pt
│   ├── fusion_model.pt
│   └── cross_attention_model.pt
│
├── results/
│   ├── plots/
│   ├── predictions/
│   ├── final_comparison.csv
│   ├── per_label_results.csv
│   ├── case_studies.csv
│   └── *.json
│
├── src/
├── notebooks/
│   └── GNN_BERT_Music_Demo.ipynb
│
├── requirements.txt
└── README.md
```

The exact model filenames may differ if checkpoints were renamed during development. The demo notebook allows checkpoint paths to be adjusted.

---

## 4. Environment

Recommended setup:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses:

- Python
- PyTorch
- PyTorch Geometric
- Hugging Face Transformers
- librosa
- NumPy
- pandas
- scikit-learn
- matplotlib

For GPU use, install a PyTorch build appropriate for the CUDA version supported by the machine before installing/using PyTorch Geometric.

---

## 5. Dataset setup

Place the FMA-small audio files under:

```text
data/raw/fma_small/
```

and FMA metadata under:

```text
data/raw/fma_metadata/
```

The metadata directory should contain at least:

```text
tracks.csv
```

The project uses the metadata's supplied split field:

```text
training
validation
test
```

---

## 6. Reproduce the pipeline

Run commands from the project root.

### Step 1 — Select the 20 target labels

```bash
cd src
python build_labels.py
```

This creates:

```text
data/splits/labels.json
```

### Step 2 — Build audio graphs

```bash
python create_graphs.py
```

The graph builder:

- divides tracks into 5-second segments;
- extracts 13 MFCC means;
- extracts 12 chroma means;
- creates temporal edges between consecutive segments;
- adds bidirectional similarity edges when cosine similarity >= 0.8.

Completed preprocessing produced 7,994 graphs.

### Step 3 — Prepare the context dataset

```bash
python prepare_context_dataset.py
```

This creates:

```text
data/splits/context_dataset.csv
```

The textual input contains:

- title
- artist
- album
- genre

The target tags are deliberately excluded from the input.

### Step 4 — Train/evaluate the models

The repository contains the individual training/evaluation scripts for the completed experiments.

The important final models are:

```text
BERT
GNN / GraphSAGE
Early Fusion
Cross-Attention Fusion
```

The corresponding saved checkpoints should be placed in:

```text
models/
```

### Step 5 — Generate evaluation artifacts

The project includes scripts for:

```text
evaluate_saved_models.py
generate_comparison.py
plot_results.py
per_label_analysis.py
tsne_analysis.py
case_studies.py
plot_cross_attention_history.py
threshold_analysis.py
```

The final artifacts are stored under:

```text
results/
```

---

## 7. Demo notebook

Open:

```text
notebooks/GNN_BERT_Music_Demo.ipynb
```

The notebook:

1. loads the final dataset and label vocabulary;
2. displays the final model comparison;
3. selects a test track;
4. loads its graph;
5. displays the graph dimensions;
6. displays saved model predictions;
7. optionally loads trained checkpoints for live inference;
8. shows the top predicted tags;
9. displays the final comparison and cross-attention plots.

This makes the notebook usable even if only the saved evaluation artifacts are available, while also supporting real checkpoint inference when the model files are present.

---

## 8. Important methodological note: target leakage

During development, an earlier BERT experiment used target tags as part of the input text.

That produced approximately:

```text
Macro-F1 = 0.450
Micro-F1 = 0.989
```

Those results are **not valid final results** because the input contained information directly related to the labels being predicted.

The experiment was corrected.

The final context input contains:

```text
Title + Artist + Album + Genre
```

while the selected 20 tags remain prediction targets only.

All results in the final comparison table use the corrected dataset.

---

## 9. Threshold note

The primary reported F1 metrics use:

```text
threshold = 0.5
```

An exploratory threshold sweep was also performed using the final test predictions. It produced higher F1 values at lower thresholds.

Those threshold-sweep values should **not** be presented as the final model performance because choosing the threshold after observing the test set introduces test-set tuning.

The official reported results therefore remain the 0.5-threshold values in the main comparison table.

---

## 10. Interpretation

### BERT

BERT performs strongly on aggregate multi-label prediction:

```text
Micro-F1 = 0.2626
Micro AUC-PR = 0.3763
```

This indicates that title/artist/album/genre context provides useful information about the selected tags.

### GNN

The GNN obtains:

```text
Macro-F1 = 0.0435
Macro AUC-PR = 0.2754
```

Its Macro AUC-PR is surprisingly competitive despite its lower F1. This suggests that the graph representation provides useful ranking information that is not fully captured by a fixed 0.5 threshold.

### Early fusion

Simply concatenating BERT and GNN embeddings does not outperform BERT:

```text
Macro-F1 = 0.0489
Micro-F1 = 0.1573
```

Combining representations therefore does not automatically make the model better.

### Cross-attention

Cross-attention produces the strongest overall ranking and macro-level performance:

```text
Macro-F1 = 0.1128
Macro AUC-PR = 0.2820
Micro AUC-PR = 0.4339
```

The model allows graph-derived representations to interact with individual BERT token representations rather than simply appending two fixed embeddings.

---

## 11. Limitations

The final results have several important limitations:

- Only 487 tracks satisfied all dataset, graph, split, and label requirements.
- The final test set contains only 50 tracks.
- The 20 labels are highly imbalanced.
- Several labels have zero positive examples in the test set.
- The graph representation uses simple MFCC/chroma averages rather than richer musical representations.
- Mean graph pooling may discard temporal information.
- The CNN baseline is a small conventional acoustic baseline and should not be interpreted as representative of modern pretrained audio models.
- t-SNE visualizations are qualitative, especially with only 50 test samples.
- The optional MusicCaps/InfoNCE retrieval extension was not implemented.
- The fixed 0.5 threshold was not optimized on the validation set.

---

## 12. Suggested submission contents

A clean final submission can contain:

```text
gnn_bert_music/
├── data/
├── models/
├── results/
├── src/
├── notebooks/
│   └── GNN_BERT_Music_Demo.ipynb
├── README.md
├── requirements.txt
└── report.pdf
```

If the complete raw FMA-small audio dataset is too large for the submission limit, provide the required processed sample graphs and document the expected dataset directory structure instead of bundling unnecessary raw audio.

---

## 13. Project conclusion

The completed experiment demonstrates a multimodal pipeline for supervised music understanding using language context and graph-based audio representations.

Cross-attention fusion is the strongest model by Macro-F1 and both AUC-PR measures, while Context BERT has the highest Micro-F1. The comparison between early concatenation and cross-attention suggests that explicit interaction between modalities is more promising than simple concatenation.

The results are exploratory because of the small usable dataset and simplified graph construction, but they provide a complete implementation and evaluation of the proposed GNN-BERT approach.
