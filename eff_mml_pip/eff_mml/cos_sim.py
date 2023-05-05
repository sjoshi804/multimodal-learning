from datetime import datetime
from typing import Dict, List
import numpy as np
import torch
import multiprocessing
import math 

class CosSim():
    def __init__(
        self,
        partition: Dict[str, List[int]],
        X: torch.tensor,
        block_size: int, 
        device_ids: int,
        run_name: str,
    ):

        # Date Time String
        DT_STRING = "".join(str(datetime.now()).split())
        
        device_idx = 0
        for latent_class in partition.keys():
            X_batch = X[partition[latent_class]].to(device_ids[device_idx])
            filename = f"{run_name}-{DT_STRING}-{latent_class}"
            multiprocessing.Process(target=self.pairwise_distance, args=(X_batch, X_batch, block_size, filename))
            device_idx = (device_idx + 1) % len(device_ids)


    def pairwise_distance(self, X1: torch.tensor, X2: torch.tensor, block_size: int, filename: str):
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
        similarity_matrix.tofile(f"{filename}.npy")
        exit()