
import streamlit as st
import pandas as pd
import os

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="OGBN-Arxiv Graph Intelligence Dashboard",
    layout="wide"
)
# -------------------------------------------------------
# Custom Dashboard Background
# -------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #e8f1ff,
            #f8fbff
        );
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #1f4e79,
            #2c7fb8
        );
    }

    /* Sidebar text color */
    [data-testid="stSidebar"] * {
        color: white;
    }

    /* Title styling */
    h1 {
        color: #0b3d91;
        font-size: 42px;
    }

    /* Headers */
    h2, h3 {
        color: #154360;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }

    </style>
    """,
    unsafe_allow_html=True
)
# -------------------------------------------------------
# Dashboard Title
# -------------------------------------------------------
st.title("OGBN-Arxiv Graph Intelligence Dashboard")

st.markdown("""
This dashboard presents the implementation and evaluation of **Graph Neural Networks (GNNs)** on the **OGBN-Arxiv** citation network dataset.

The dashboard includes:

- Graph Statistics
- Model Performance
- Node Classification Results
- Embedding Visualizations
- Feature Importance
""")

st.markdown("---")

# -------------------------------------------------------
# Base results directory (change this if your project folder differs)
# -------------------------------------------------------
BASE_DIR = "/content/drive/MyDrive/CCS4354_Tensors_and_Graphs/results"

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "Home",
        "Dataset",
        "Model Performance",
        "Embeddings",
        "Feature Importance",
        "Bonus & Optimization",
        "About"
    ]
)

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------

if page == "Home":

    st.subheader(
        "Node Classification Using Graph Neural Networks on the OGBN-Arxiv Citation Network"
    )

    st.markdown("---")

    # ---------------------------------------------------
    # Student Information
    # ---------------------------------------------------

    st.header("Student Group")

    student_data = {
        "Student Name": [
            "Amintha Jayasooriya",
            "Tharanya Pushparaj",
            "Damsara Dissanayaka",
            "Thamindu Kavinda",
            "Ravindi Ayodhya"
        ],
        "Student ID": [
            "CIT-23-02-0335",
            "CIT-23-02-0176",
            "CIT-23-02-0163",
            "CIT-23-02-0356",
            "23UG1-0136"
        ]
    }

    st.table(student_data)

    st.markdown("---")

    # ---------------------------------------------------
    # Module Information
    # ---------------------------------------------------

    st.header("Module Information")

    col1, col2 = st.columns(2)

    with col1:
        st.info("**Module Code**\n\nCCS4354")
        st.info("**Module Name**\n\nTensors and Graphs")

    with col2:
        st.info("**Supervisor**\n\nDr. Chameera De Silva")
        st.info("**Dataset**\n\nOGBN-Arxiv")

    st.markdown("---")

    # ---------------------------------------------------
    # Dataset Overview
    # ---------------------------------------------------

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(label="Nodes", value="169,343")
    col2.metric(label="Edges", value="1,166,243")
    col3.metric(label="Node Features", value="128")
    col4.metric(label="Classes", value="40")

    st.markdown("---")

    # ---------------------------------------------------
    # Technologies Used
    # ---------------------------------------------------

    st.header("Technologies Used")

    tech1, tech2 = st.columns(2)

    with tech1:
        st.success("PyTorch")
        st.success("PyTorch Geometric")
        st.success("Open Graph Benchmark (OGB)")
        st.success("NetworkX")

    with tech2:
        st.success("Streamlit")
        st.success("Scikit-learn")
        st.success("Matplotlib")
        st.success("Google Colab")

    st.markdown("---")

    st.success("Graph Intelligence Dashboard developed for CCS4354 - Tensors and Graphs")

# -------------------------------------------------------
# DATASET PAGE
# -------------------------------------------------------

elif page == "Dataset":

    st.header("Graph Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Nodes", "169,343")
    col2.metric("Edges", "1,166,243")
    col3.metric("Features", "128")
    col4.metric("Classes", "40")

    st.markdown("---")

    st.subheader("Dataset Information")

    st.write("""
The **OGBN-Arxiv** dataset is a citation network where:

- Nodes represent research papers.
- Edges represent citation relationships.
- Each node contains 128 numerical features.
- The task is node classification into 40 research categories.
""")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # -----------------------------------------------
    # Left Column - Sample Graph
    # -----------------------------------------------
    with col1:

        st.subheader("Sample Subgraph")

        image_path = os.path.join(BASE_DIR, "graphs", "sample_subgraph.png")

        if os.path.exists(image_path):
            st.image(
                image_path,
                caption="Sample Subgraph of the OGBN-Arxiv Citation Network",
                use_container_width=True
            )
        else:
            st.warning("sample_subgraph.png not found.")

    # -----------------------------------------------
    # Right Column - Degree Distribution
    # -----------------------------------------------
    with col2:

        st.subheader("Degree Distribution")

        # Histogram
        image_path = os.path.join(BASE_DIR, "plots", "degree_distribution.png")

        if os.path.exists(image_path):
            st.image(
                image_path,
                caption="Histogram",
                use_container_width=True
            )
        else:
            st.warning("degree_distribution.png not found.")

        st.markdown("#### Degree Rank Plot")

        # Degree Rank Plot
        image_path = os.path.join(BASE_DIR, "plots", "degree_rank_distribution.png")

        if os.path.exists(image_path):
            st.image(
                image_path,
                caption="Degree Rank Plot",
                use_container_width=True
            )
        else:
            st.warning("degree_rank_distribution.png not found.")

# -------------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance")

    st.subheader("Model Comparison")

    eval_path = os.path.join(BASE_DIR, "evaluation", "model_comparison.csv")

    if os.path.exists(eval_path):
        df = pd.read_csv(eval_path)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("model_comparison.csv not found.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Loss")
        image_path = os.path.join(BASE_DIR, "plots", "training_loss_comparison.png")
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning("training_loss_comparison.png not found.")

    with col2:
        st.subheader("Validation Accuracy")
        image_path = os.path.join(BASE_DIR, "plots", "validation_accuracy_comparison.png")
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning("validation_accuracy_comparison.png not found.")

    st.markdown("---")

    st.subheader("Validation Macro-F1")
    image_path = os.path.join(BASE_DIR, "plots", "validation_f1_comparison.png")
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("validation_f1_comparison.png not found.")

    st.markdown("---")

    st.success("GraphSAGE achieved better performance than GCN on the OGBN-Arxiv dataset.")

# -------------------------------------------------------
# EMBEDDINGS
# -------------------------------------------------------

elif page == "Embeddings":

    st.header("Embedding Visualizations")

    st.write("""
The learned node embeddings were visualized using
Principal Component Analysis (PCA) and t-SNE.
""")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("PCA Embedding (GCN)")
        image_path = os.path.join(BASE_DIR, "plots", "gcn_embedding_pca.png")
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning("gcn_embedding_pca.png not found.")

    with col2:
        st.subheader("t-SNE Embedding (GCN)")
        image_path = os.path.join(BASE_DIR, "plots", "gcn_embedding_tsne.png")
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning("gcn_embedding_tsne.png not found.")

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("PCA Embedding (GraphSAGE)")
        image_path = os.path.join(BASE_DIR, "plots", "sage_embedding_pca.png")
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning("sage_embedding_pca.png not found.")

    with col4:
        st.subheader("t-SNE Embedding (GraphSAGE)")
        image_path = os.path.join(BASE_DIR, "plots", "sage_embedding_tsne.png")
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning("sage_embedding_tsne.png not found.")

# -------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------

elif page == "Feature Importance":

    st.header("Feature Importance")

    st.write("""
The following figure shows the Top 10 important node features
used by the Graph Neural Network.
""")

    image_path = os.path.join(BASE_DIR, "plots", "feature_importance_top10.png")

    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("feature_importance_top10.png not found.")

# -------------------------------------------------------
# ABOUT
# -------------------------------------------------------

elif page == "Bonus & Optimization":

    st.header("Bonus Extensions & Performance Optimization")

    st.write("""
This page covers the additional work built on top of the required Tasks 01-08:
the Bonus architectures (Graph Transformer, Relational GNN, Self-Supervised
DGI pretraining, GNNExplainer) and the Performance Optimization techniques
(Focal Loss, weighted ensembling, Correct-and-Smooth).
""")

    st.markdown("---")

    # -----------------------------------------------
    # Bonus models comparison
    # -----------------------------------------------
    st.subheader("Bonus Models Comparison (Test Set)")

    csv_path = os.path.join(BASE_DIR, "evaluation", "bonus_model_comparison.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("bonus_model_comparison.csv not found.")

    image_path = os.path.join(BASE_DIR, "plots", "bonus_architecture_comparison.png")
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("bonus_architecture_comparison.png not found.")

    st.markdown("---")

    # -----------------------------------------------
    # DGI + GNNExplainer
    # -----------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Self-Supervised (DGI) Pretraining Loss")
        image_path = os.path.join(BASE_DIR, "plots", "dgi_pretraining_loss.png")
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning("dgi_pretraining_loss.png not found.")

    with col2:
        st.subheader("GNNExplainer - Edge Importance")
        image_path = os.path.join(BASE_DIR, "plots", "gnnexplainer_node_explanation.png")
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning("gnnexplainer_node_explanation.png not found.")

    st.markdown("---")

    # -----------------------------------------------
    # Performance optimization
    # -----------------------------------------------
    st.subheader("Performance Optimization - Every Model (Test Set)")

    csv_path = os.path.join(BASE_DIR, "evaluation", "performance_optimization_comparison.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("performance_optimization_comparison.csv not found.")

    image_path = os.path.join(BASE_DIR, "plots", "performance_optimization_comparison.png")
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("performance_optimization_comparison.png not found.")

    st.markdown("---")

    st.subheader("Weighted Ensemble - Model Weights")
    csv_path = os.path.join(BASE_DIR, "evaluation", "ensemble_weights.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("ensemble_weights.csv not found.")

    st.markdown("---")

    st.success("Bonus extensions and optimization techniques improved on the required GCN/GraphSAGE baseline.")

elif page == "About":

    st.header("About This Project")

    st.markdown("""
### Project

Graph Intelligence using Graph Neural Networks

---

### Dataset

- OGBN-Arxiv
- Citation Network Dataset

---

### Implemented Models

- Graph Convolutional Network (GCN)
- GraphSAGE

---

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

---

### Frameworks

- PyTorch
- PyTorch Geometric
- OGB
- Streamlit

---

### Developed For

CCS4354 - Tensors and Graphs

BSc (Hons) Data Science
""")

    st.success("Dashboard completed successfully.")
