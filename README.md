# Analysing Modality Gap Reduction Using Spectral Embeddings for Image Retrieval

This project investigates whether **spectral embeddings** can reduce the modality gap in image-text retrieval models. It evaluates the impact of spectral post-processing on shared embedding spaces produced by models such as **CLIP**, **JinaCLIP**, and **LongCLIP**.

The study focuses on cross-modal retrieval tasks using the **PixelProse dataset**, comparing model performance before and after applying spectral embedding transformations.

## Overview

Image-text models map images and text into a shared embedding space for retrieval tasks. However, these embeddings often suffer from a **modality gap**, where image embeddings and text embeddings form separate clusters.

This project explores whether spectral embeddings can improve alignment between modalities by restructuring the embedding space.

The main goals are:

- Evaluate baseline retrieval performance of CLIP, JinaCLIP, and LongCLIP
- Apply spectral embedding as a post-processing step
- Compare retrieval performance before and after transformation
- Visualize embedding spaces using PCA and UMAP
- Analyze whether improved global alignment leads to better retrieval accuracy

## Models Used

The following multimodal models are evaluated:

- **CLIP**
- **JinaCLIP**
- **LongCLIP**

Each model is tested using both short and long captions where applicable.

## Dataset

The experiments are conducted on the **PixelProse dataset**, which contains image-caption pairs suitable for image-text retrieval evaluation.

## Methodology

The workflow consists of two main pipelines:

### Baseline Pipeline

1. Load images and captions
2. Generate image and text embeddings using a multimodal model
3. Store image embeddings in a vector database
4. Retrieve nearest matches using cosine similarity
5. Evaluate retrieval performance

### Spectral Embedding Pipeline

1. Generate original image and text embeddings
2. Construct an adjacency matrix based on embedding similarity
3. Compute the degree matrix
4. Build the graph Laplacian
5. Perform eigen-decomposition
6. Transform embeddings into a spectral space
7. Evaluate retrieval metrics again

## Spectral Embedding Formulation

The spectral transformation is based on graph Laplacian eigen-decomposition.

### Adjacency Matrix

```math
A_{ij} =
\begin{cases}
w_{ij}, & \text{if node } i \text{ and node } j \text{ are connected} \\
0, & \text{otherwise}
\end{cases}
````

### Degree Matrix

```math
D_{ii} = \sum_{j=1}^{n} A_{ij}
```

### Laplacian Matrix

```math
L = D - A
```

### Eigen-Decomposition

```math
Lv = \lambda v
```

The resulting eigenvectors are used to form the spectral embedding space.

## Key Results

Spectral embeddings successfully reduced the global modality gap, as observed in PCA and UMAP visualizations. However, retrieval performance did not consistently improve.

## Results Summary

| Model    | Caption Type |    Metric | Before |  After |
| -------- | ------------ | --------: | -----: | -----: |
| CLIP     | Short        |       MRR | 0.8696 | 0.7129 |
| CLIP     | Short        |  Recall@1 | 0.8238 | 0.6313 |
| CLIP     | Short        |  Recall@5 | 0.9300 | 0.8325 |
| CLIP     | Short        | Recall@10 | 0.9550 | 0.8875 |
| JinaCLIP | Short        |       MRR | 0.8525 | 0.6797 |
| JinaCLIP | Short        |  Recall@1 | 0.7988 | 0.5863 |
| JinaCLIP | Short        |  Recall@5 | 0.9250 | 0.8163 |
| JinaCLIP | Short        | Recall@10 | 0.9563 | 0.8650 |
| LongCLIP | Short        |       MRR | 0.8536 | 0.7912 |
| LongCLIP | Short        |  Recall@1 | 0.8038 | 0.7250 |
| LongCLIP | Short        |  Recall@5 | 0.9163 | 0.8838 |
| LongCLIP | Short        | Recall@10 | 0.9425 | 0.9200 |

## Project Structure

```text
.
├── data/
│   └── pixelprose/
├── embeddings/
│   ├── original/
│   └── spectral/
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── generate_embeddings.py
│   ├── spectral_embedding.py
│   ├── retrieval.py
│   ├── metrics.py
│   └── visualization.py
├── results/
│   ├── figures/
│   ├── heatmaps/
│   └── metrics/
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/modality-gap-spectral-embeddings.git
cd modality-gap-spectral-embeddings
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Embeddings

```bash
python src/generate_embeddings.py --model clip --dataset data/pixelprose
```

### 2. Apply Spectral Embedding

```bash
python src/spectral_embedding.py --input embeddings/original --output embeddings/spectral
```

### 3. Run Retrieval Evaluation

```bash
python src/retrieval.py --embeddings embeddings/spectral
```

### 4. Generate Visualizations

```bash
python src/visualization.py --embeddings embeddings/spectral --output results/figures
```

## Visualizations

The project includes visual analysis using:

* PCA plots before and after spectral transformation
* UMAP plots before and after spectral transformation
* Cross-modal image-text heatmaps
* Intra-modal image-image and text-text heatmaps
* Eigenvalue decay plots

These visualizations help analyze whether image and text embeddings become more aligned after spectral transformation.

## Conclusion

This project shows that spectral embeddings can reduce the global modality gap in image-text embedding spaces. However, reducing the modality gap alone is not sufficient to improve retrieval accuracy.

Although spectral transformation improves global alignment and locality preservation, it may distort fine-grained neighborhood relationships that are critical for accurate cross-modal retrieval.

LongCLIP showed the most stable behavior after spectral transformation, suggesting that longer-caption models may better preserve semantic structure under spectral post-processing.


## References

1. François Role, Sebastien Meyer, and Victor Amblard. *Fill the Gap: Quantifying and Reducing the Modality Gap in Image-Text Representation Learning*. 2025.
2. Alec Radford et al. *Learning Transferable Visual Models from Natural Language Supervision*. 2021.
3. Han Xiao et al. *Jina CLIP: Your CLIP Model Is Also Your Text Retriever*. 2024.
4. Xinyang Zhai et al. *Sigmoid Loss for Language Image Pre-Training*. 2023.
5. Leland McInnes, John Healy, and James Melville. *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. 2018.
6. Jonathon Shlens. *A Tutorial on Principal Component Analysis*. 2014.
