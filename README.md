# DEMEC: Drug Embedding & Multi-Effect Classification

## Overview
Assessing drug side effects represents a critical challenge for clinicians weighing a drug’s benefits against its risks and for pharmaceutical companies developing novel therapeutics. However, accurate predictions remain challenging because small changes in the chemical structure or properties of a drug can precipitate new side effects, and the number of possible side effects ranges in the thousands.  

DEMEC (Drug Embedding & Multi-Effect Classification) aims to leverage **Graph Neural Networks (GNNs)** to address these challenges by learning rich molecular embeddings and performing **multi-label classification** of side effects.

---

## Datasets

### SIDER 4.1 (Side Effect Resource)
- Source: [http://sideeffects.embl.de/](http://sideeffects.embl.de/)
- Provides information about 5,868 side effects, 1,430 drugs, and 139,756 drug–side-effect interactions.
- Includes ATC codes that identify and group drugs based on pharmacological and therapeutic properties.

### DrugBank Database
- Source: [https://pubmed.ncbi.nlm.nih.gov/29126136/](https://pubmed.ncbi.nlm.nih.gov/29126136/)
- Provides detailed drug chemical properties and molecular structures.
- Used to link chemical structure information to the SIDER dataset via CID/ATC codes.

These datasets together allow us to build a graph representation that connects molecular substructures, drugs, and their associated side effects.

### Downloading Data

This repository includes **processed and aggregated data** from the **SIDER** and **DrugBank** databases, located under:

```
data/processed/
├─ smiles_cache.csv              # Cached SMILES for each CID
└─ graphs/                       # Contains one [CID].gpickle file per drug
```

If you would like to download and process the **raw SIDER dataset** yourself, run the following commands from the repository root:

```bash
cd data
curl -O http://sideeffects.embl.de/media/download/README
curl -O http://sideeffects.embl.de/media/download/drug_names.tsv
curl -O http://sideeffects.embl.de/media/download/drug_atc.tsv
curl -O http://sideeffects.embl.de/media/download/medra_all_indications.tsv.gz
curl -O http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz
curl -O http://sideeffects.embl.de/media/download/meddra_freq.tsv.gz
gunzip *.gz
```

After downloading, you can **reprocess and aggregate** the data into molecular graphs and cached SMILES by running:

```bash
python scripts/aggregate_data.py data/drug_names.tsv data/drug_atc.tsv
```

This will populate `data/processed/smiles_cache.csv` and `data/processed/graphs/` with the processed results.

---

## Problem Definition

### Goal
Given a **novel drug** and its **chemical structure**, predict which **side effects** it may cause in patients.

### Metric
- **Primary:** ROC-AUC (multi-class, one-vs-rest)
- **Secondary:** True Positive Rate (TPR) and False Positive Rate (FPR) per side effect to analyze under- or over-prediction.

### Why These Datasets
- **SIDER**: High-quality, well-documented dataset with drug–side-effect mappings and ATC codes for integration.
- **DrugBank**: Industry-standard resource linking structure and properties; allows transfer learning between datasets.

---

## Graph Representation
Each **drug molecule** is represented as a graph:
- **Nodes:** Atoms (features such as atom type, degree, formal charge, hybridization, aromaticity)
- **Edges:** Chemical bonds (features include bond type and stereochemistry)

Connections are also drawn between drugs and side effects or between drugs that share molecular substructures, forming a rich relational network for learning.

---

## Model Architecture

### Encoder
A **Graph Neural Network (GNN)** serves as the encoder, learning an embedding for each molecule:
- **Baseline:** Graph Convolutional Network (GCN)
- **Advanced Models:** Graph Attention Network (GAT), Attentive FP, and Message Passing Neural Network (MPNN) (if time permits)

### Multi-Task Prediction Heads
Each embedding is fed into multiple classification heads:
1. **Side Effect Prediction (Primary Task)** – multi-label binary classification (≈5,000 outputs)
2. **ATC Classification** – multi-label classification (14 outputs)
3. **FDA Approval Status** – binary classification
4. **Drug-Likeness (Lipinski’s Rule)** – binary classification
5. **BBB Permeability** – binary classification
6. **Hepatotoxicity** – binary classification

### Loss Function
$$
L_{total} = L_{side\ effects} + \sum_i \lambda_i L_{auxiliary_i}
$$
Each loss term is a binary cross-entropy loss. The weights $\lambda_i$ balance auxiliary tasks, either uniformly (0.2) or through learned uncertainty-based weighting.

---

## Motivation for GNNs and Multi-Task Learning
- **Graph Structure Fit:** Molecules are naturally graph-structured, making GNNs ideal for representing atomic connectivity and chemical context.  
- **Multi-Task Synergy:** Predicting related drug properties (ATC class, toxicity, permeability) helps the model learn more meaningful molecular embeddings that generalize to unseen drugs.

---

## Novelty
Previous approaches often required explicit molecular property annotations as model inputs, which are expensive to determine. DEMEC introduces these as **auxiliary losses** during training, enabling prediction from structure alone. This approach encourages the GNN to learn chemically robust representations even when auxiliary labels are missing.

---

## Planned Experiments
- **Baseline:** GCN with a single side-effect classification head  
- **Multi-Task:** Add auxiliary prediction heads with masked losses  
- **Ablations:** Remove individual heads and compare performance  
- **Advanced Models:** Evaluate GAT and Attentive FP for improved aggregation

---

## Metrics & Evaluation
- ROC-AUC (macro/micro)
- TPR/FPR per side effect
- Embedding visualization (t-SNE) for molecular representation quality

---

## References
- **SIDER 4.1:** Kuhn et al., *Nucleic Acids Research* (2016). [http://sideeffects.embl.de/](http://sideeffects.embl.de/)  
- **DrugBank:** Wishart et al., *Nucleic Acids Research* (2018). [https://pubmed.ncbi.nlm.nih.gov/29126136/](https://pubmed.ncbi.nlm.nih.gov/29126136/)  
- **Multi-task GNNs for Molecular Prediction:** [https://pmc.ncbi.nlm.nih.gov/articles/PMC11606038/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11606038/)
