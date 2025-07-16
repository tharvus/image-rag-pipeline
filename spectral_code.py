import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

def compute_spectral_embedding(embeddings, n_components):
    """
    Compute spectral embedding using normalized graph Laplacian eigenvectors.
    embeddings: numpy array (N x D)
    n_components: number of eigenvectors to return
    """
    # Similarity matrix (cosine similarity)
    similarity = cosine_similarity(embeddings)

    similarity[similarity < 0] = 0
    np.fill_diagonal(similarity, 1.0 + 1e-9)
    

    # Compute normalized Laplacian
    L = laplacian(similarity, normed=True)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = eigh(L)
    
    # Return eigenvectors corresponding to 2nd to (n_components+1)-th smallest eigenvalues
    return eigvecs[:, 1:n_components+1]