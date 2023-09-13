import numpy as np
import torch

def compute_cosine_similarity(X1: torch.tensor, X2: torch.tensor, block_size: int, filename: str):
    similarity_matrices = []
    for i in range(X1.shape[0] // block_size + 1):
        similarity_matrices_i = []
        e = X1[i*block_size:(i+1)*block_size]
        for j in range(X2.shape[0] // block_size + 1):
            e_t = X2[j*block_size:(j+1)*block_size].t()
            similarity_matrices_i.append(
                np.array(
                torch.cosine_similarity(e[:, :, None], e_t[None, :, :]).detach().cpu()
                )
            )
        similarity_matrices.append(similarity_matrices_i)
    similarity_matrix = np.block(similarity_matrices).astype(np.float16)

    with open(f"{filename}.npy", "wb") as f:
        np.save(f, similarity_matrix)