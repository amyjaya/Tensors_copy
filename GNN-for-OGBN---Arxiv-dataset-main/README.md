# 🧠 OGBN-Arxiv Graph Neural Network — Node Classification
CCS4354 Tensors and Graphs Coursework | SLTC | Faculty of Computing and Information Technology

An end-to-end Graph Neural Network pipeline for multi-class node classification on the Open Graph Benchmark's ogbn-arxiv citation network — comparing GCN and GraphSAGE architectures, with full explainability analysis (PCA, t-SNE, neighbourhood influence, feature importance) and an interactive Streamlit Graph Intelligence Dashboard.

## 📋 Table of Contents
- Problem Statement
- Dataset
- Technologies & Tools
- Tensor Fundamentals (Task 01)
- Graph Representation & Analysis (Task 02)
- Data Preparation (Task 03)
- GNN Architectures (Task 04)
- Training & Hyperparameter Tuning (Task 05)
- Model Evaluation (Task 06)
- Explainability Analysis (Task 07)
- Graph Intelligence Dashboard (Task 08)
- Results Summary
- Challenges & Limitations
- Future Improvements
- Project Structure
- Getting Started
- Team

## 🎯 Problem Statement
Predict the research field of scientific papers using their content features and citation relationships in the ogbn-arxiv citation network.

| Item | Specification |
|---|---|
| Task | Semi-supervised, multi-class node classification |
| Inputs | 128-dimensional node feature vectors (title + abstract embeddings), citation graph |
| Outputs | One of 40 arXiv Computer Science subject categories per paper |

## 📊 Dataset
| Property | Value |
|---|---|
| Source | Open Graph Benchmark (OGB), via `PygNodePropPredDataset` |
| Nodes | 169,343 scientific papers |
| Edges | 1,166,243 directed citation edges (1,157,799 unique undirected edges) |
| Node Features | 128-dimensional skip-gram embeddings of title + abstract |
| Labels | 40 arXiv CS subject categories (e.g. cs.LG, cs.CV, cs.AI) |
| Average Degree | 13.67 · Maximum Degree | 13,161 |
| Graph Density | 8.07 × 10⁻⁵ (very sparse) |
| Connected Components | 1 (all 169,343 nodes in a single component) |

**Official OGB Split:**

| Split | Nodes |
|---|---|
| Training | 90,941 |
| Validation | 29,799 |
| Test | 48,603 |

## 🛠 Technologies & Tools
- **Language:** Python 3.12, on Google Colab with a Tesla T4 GPU
- **Frameworks:** PyTorch (tensors, autograd, GPU training), PyTorch Geometric (`GCNConv`, `SAGEConv`, `Data` objects)
- **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn (StandardScaler, PCA, t-SNE, metrics), NetworkX (structural graph analysis), Streamlit (dashboard)

## 🔢 Tensor Fundamentals (Task 01)
Core PyTorch tensor operations were demonstrated as the foundation for all later GNN computations:
- **Creation:** scalars, vectors, matrices, random tensors, NumPy conversion
- **Indexing:** extracting node rows, individual values, and slices — mirroring how node feature rows are read from the ogbn-arxiv feature matrix
- **Reshaping:** `.reshape()` / `.flatten()` for preparing feature arrays for graph convolution layers
- **Matrix Multiplication:** `torch.matmul`, the core operation inside every GCN/GraphSAGE linear layer
- **Broadcasting:** vector + matrix addition, as used when adding bias vectors to node embeddings
- **Aggregation:** sum, mean, max, min, row-wise sum
- **GPU Operations:** verified CUDA acceleration on a Tesla T4, moving tensors from CPU to GPU with `.to(device)`

## 🕸 Graph Representation & Analysis (Task 02)
- Loaded the graph via `PygNodePropPredDataset` as a PyG `Data` object (edge index, node features, node years, labels), then converted the edge index into a NetworkX-compatible edge-list DataFrame.
- **Degree Distribution:** heavy-tailed, power-law-like — most papers have low degree, a few are highly cited (max degree 13,161).
- **Density Analysis:** ~0.008% density confirms an extremely sparse graph, motivating sparse-aware GNN operations.
- **Connected Components:** exactly 1 component containing all nodes — confirms message passing can, in principle, reach any node in the graph.

## 🔧 Data Preparation (Task 03)
- Loaded the 169,343 × 128 node feature matrix and 169,343 × 1 label tensor directly from the PyG dataset.
- Used the **official OGB train/validation/test split** (by publication year) for standardised, leakage-free benchmarking.
- **Feature normalisation:** `StandardScaler` fit strictly on training nodes only, then applied to the full feature matrix — preventing validation/test leakage.
- Retained PyTorch Geometric's `Data` format throughout for efficient large-scale graph handling.

```
Raw Dataset → Feature Extraction → Normalization → Train / Validation / Test Split → Model Training
```

## 🤖 GNN Architectures (Task 04)
Two 2-layer GNNs were implemented using PyTorch Geometric convolution layers:

**Model 1 — Graph Convolutional Network (GCN)**
- Degree-normalised weighted aggregation of neighbour features (including self-loops), followed by a learnable linear transform
- Hidden dimension: 256 · Activation: ReLU · Dropout: 0.5 · Layers: 2 · Trainable parameters: 43,304

```
Input Features (128-d) → GCN Layer (128→256) → ReLU → Dropout (0.5) → GCN Layer (256→40) → Classification Output (40 classes)
```

**Model 2 — GraphSAGE**
- Learns to aggregate neighbourhood features via an explicit aggregator (mean, in this implementation), combined with the node's own features through a learnable transform
- Hidden dimension: 256 · Dropout: 0.5 · Layers: 2 · Trainable parameters: 86,312 (roughly double GCN's, since SAGEConv learns separate weight matrices for self vs. neighbour features)

## 🏋️ Training & Hyperparameter Tuning (Task 05)
- **Loss:** Cross-Entropy Loss (multi-class, single-label problem)
- **Optimizer:** Adam, learning rate 0.01, weight decay 5e-4, 100 epochs (baseline)
- A hyperparameter search across 4 configurations (learning rate, hidden dimension, dropout) identified the best-performing setup for both models: **learning rate 0.005, hidden dimension 256, dropout 0.3**, used for all Task 06 evaluation results.

| Experiment | LR | Hidden | Dropout | Epochs | GCN Val Acc | GCN Val F1 | SAGE Val Acc | SAGE Val F1 |
|---|---|---|---|---|---|---|---|---|
| Best | 0.005 | 256 | 0.3 | 60 | 0.7051 | 0.5328 | 0.6987 | 0.5194 |

## 📈 Model Evaluation (Task 06)
Evaluated on validation and test sets using macro-averaged Accuracy, Precision, Recall, and F1 (via scikit-learn):

| Model | Dataset | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| GCN | Validation | 0.7041 | 0.524 | 0.5537 | 0.5328 |
| **GCN** | **Test** | **0.6962** | **0.5146** | **0.5379** | **0.5169** |
| GraphSAGE | Validation | 0.6986 | 0.518 | 0.5349 | 0.5194 |
| GraphSAGE | Test | 0.6895 | 0.5131 | 0.5339 | 0.5112 |

**Best model: GCN** — outperformed GraphSAGE across every metric on both validation and test sets. Its degree-normalised aggregation provides a more restricted, low-variance update that fits the citation graph's topology well.

Macro-averaged scores are notably lower than accuracy because the 40 classes are heavily imbalanced (e.g. class 16 has 10,477 test examples; some classes have fewer than 10).

**GCN** — Strengths: simple, efficient, strong baseline on citation networks. Weaknesses: needs the full graph at training time, less scalable, prone to over-smoothing at depth.
**GraphSAGE** — Strengths: designed for large-scale/inductive learning, more scalable in principle. Weaknesses: mini-batch sampling can lose neighbourhood information (note: both models were trained full-batch here, so this didn't affect these results).

## 🔍 Explainability Analysis (Task 07)
Three complementary techniques were applied to interpret the trained GCN:

- **Embedding Visualisation (PCA & t-SNE):** 256-dimensional hidden-layer embeddings for 3,000 sampled nodes, projected to 2D and coloured by true class. Both methods show papers from similar research areas forming visible clusters — evidence the model learned meaningful representations from both features and graph structure.
- **Neighbourhood Influence Analysis:** examined the highest-degree node (node 1353, degree 13,161) and its neighbours. The majority of neighbours belonged to classes 24 and 13; the model correctly predicted class 24, demonstrating the effect of neighbour-aggregated messages.
- **Feature Importance Analysis:** ranked the 128 input feature dimensions by mean-absolute first-layer GCN weight magnitude, identifying the top 10 most influential embedding dimensions. Noted limitation: this reflects average layer-wide influence, not per-prediction effects — permutation importance, Integrated Gradients, or GNNExplainer are recommended as stronger future approaches.

## 📊 Graph Intelligence Dashboard (Task 08)
An interactive **Streamlit dashboard** (`dashboard/app.py`) was built with six pages, navigable via a sidebar:

| Page | Contents |
|---|---|
| **Home** | Module info, team table, headline dataset metrics (169,343 nodes, 1,166,243 edges, 128 features, 40 classes), tech stack summary |
| **Dataset** | Graph statistics, sample subgraph visualisation, degree distribution & degree-rank plots |
| **Model Performance** | Interactive GCN vs. GraphSAGE comparison table, training loss & validation accuracy curves |
| **Embeddings** | Side-by-side PCA and t-SNE projections of learned node embeddings |
| **Feature Importance** | Top influential input feature dimensions |
| **Bonus & Optimization** | Comparison of all bonus/optimisation model variants |
| **About** | Project summary and implemented models |

The dashboard reads pre-computed artefacts (images, model comparison CSV, saved weights) from a results directory rather than recomputing at runtime, keeping it fast and responsive.

## 🏆 Results Summary

Consolidated test-set comparison across baseline, tuned, and bonus/optimisation models:

| Category | Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| Baseline (Task 04) | GCN | 0.6966 | 0.5102 | 0.5377 | 0.5119 |
| Baseline (Task 04) | GraphSAGE | 0.6888 | 0.5116 | 0.5306 | 0.5109 |
| **Tuned (Task 06)** | **GCN** | **0.6962** | **0.5146** | **0.5379** | **0.5169** |
| Tuned (Task 06) | GraphSAGE | 0.6895 | 0.5131 | 0.5339 | 0.5112 |
| Bonus A | Graph Transformer | 0.6636 | 0.4686 | 0.4888 | 0.4635 |
| Bonus B | Relational GNN / KG-style | 0.6555 | 0.4681 | 0.5174 | 0.4817 |
| Bonus C | Self-Supervised (DGI) + GCN | 0.6992 | 0.5118 | 0.5353 | 0.5135 |
| Optimization 1 | Focal Loss + SSL GCN | 0.6944 | 0.5052 | 0.5332 | 0.5077 |
| **Optimization 2** | **Weighted 5-Model Ensemble** | 0.6959 | **0.5240** | **0.5384** | **0.5199** |

- **Highest accuracy:** Self-Supervised DGI + GCN (69.92%)
- **Highest F1:** Weighted 5-Model Ensemble (0.5199)
- Both GCN and GraphSAGE substantially outperform the feature-only MLP baseline on the official OGB leaderboard (~55.5% accuracy), confirming that citation structure carries real predictive signal beyond paper content alone.

## ⚠️ Challenges & Limitations
- **Scale:** 169,343 nodes and 1.1M+ edges required careful, efficient computation even for basic operations like degree distribution.
- **Class imbalance:** 40 categories ranging from <10 to >10,000 examples drove a large gap between accuracy and macro-averaged metrics.
- **Compute:** full-batch training of both models over 100 epochs required GPU acceleration (Tesla T4).
- **Training approach:** full-batch (not mini-batch/neighbour sampling) was used for both models — less scalable, and doesn't showcase GraphSAGE's sampling strength.
- **Overfitting signs:** validation accuracy plateaued around epoch 60–70 despite dropout (0.5) and weight decay (5e-4).

## 🚀 Future Improvements
- **Graph Transformers** — attention-based weighting of neighbours/distant nodes
- **Advanced architectures** — GCNII or Graph Attention Networks to mitigate over-smoothing
- **Self-supervised pretraining** — contrastive or masked-feature objectives before fine-tuning, especially to help under-represented classes
- **Knowledge graph integration** — enriching the citation graph with author/venue metadata for extra predictive signal

## 📁 Project Structure
```
OGBN-Arxiv-GNN/
├── README.md                          # Project overview & documentation
├── requirements.txt                   # Environment dependencies
├── notebook/
│   └── OGBN_Arxiv_GNN.ipynb           # Fully executed coursework notebook (Tasks 01–08 + Bonus)
├── report/
│   └── CCS4354_Final_Assignment_Group3.pdf   # Formal technical report
├── models/
│   ├── gcn_model.pt                   # Final selected model (best performer)
│   └── sage_model.pt
├── dashboard/
│   └── app.py                         # Streamlit Graph Intelligence Dashboard
└── results/
    ├── model_comparison.csv
    ├── degree_distribution.png
    ├── training_loss_comparison.png
    ├── validation_accuracy_comparison.png
    ├── pca_embedding.png
    ├── tsne_embedding.png
    └── feature_importance.png
```

## 🚀 Getting Started
```bash
git clone https://github.com/<your-username>/OGBN-Arxiv-GNN.git
cd OGBN-Arxiv-GNN
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## 👥 Team — CCS4354 Tensors and Graphs

| Member | Student ID | Contributions |
|---|---|---|
| **Ravindi Ayodhya** | 23UG1-0136 | Environment & dataset setup; Task 01 (Tensor Fundamentals); Bonus A (Graph Transformer); Report writing |
| **Amintha Jayasooriya** | CIT-23-02-0335 | Task 02 (Graph Representation & Analysis); Bonus B (Relational GNN / Knowledge Graph Extensions); Task 08 (Streamlit Dashboard) |
| **Tharanya Pushparaj** | CIT-23-02-0176 | Task 03 (Graph Data Preparation); Bonus C (Self-Supervised Pretraining); Task 08 (Streamlit Dashboard) |
| **Damsara Dissanayaka** | CIT-23-02-0163 | Task 04 (GNN Development); Task 05 (Model Training & Optimization); Bonus D (GNNExplainer); Performance optimization |
| **Thamindu Kavinda** | CIT-23-02-0356 | Task 06 (Model Evaluation); Task 07 (Graph Explainability) |

⚠️ **Academic Disclaimer:** This project was developed for academic evaluation as part of the CCS4354 Tensors and Graphs module at Sri Lanka Technology Campus.
