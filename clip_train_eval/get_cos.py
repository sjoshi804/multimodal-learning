from abc import ABC, abstractmethod
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
import clip
import pandas as pd
from tqdm.notebook import tqdm
from pkg_resources import packaging
from PIL import Image 
from tqdm import tqdm 
import torch
import pickle
import matplotlib.pyplot as plt 

block_size = 512
similarity_matrices = []
X1 = torch.load('full_text_embeds2.pt', map_location='cuda')
X2 = torch.load('full_image_embeds2.pt', map_location='cuda')
for i in tqdm(range(X1.shape[0] // block_size + 1)):
    similarity_matrices_i = []
    e = X1[i*block_size:(i+1)*block_size]
    e_t = X2[i*block_size:(i+1)*block_size]
    similarity_matrices.append(
        np.array(
        torch.cosine_similarity(e,e_t).detach().cpu()
        )
    )
    
similarity_matrix = np.block(similarity_matrices).astype(np.float16)
sorted_matrix = np.sort(similarity_matrix)
'''
plt.plot(sorted_matrix.tolist())
plt.ylabel("Cos Similarity")
plt.title("Cosine Similarity Distribution")
plt.xlabel('Index')
#plt.xlim(1000,200000)
plt.savefig('cos_full2.png')

'''
inds = similarity_matrix.argsort()
f = open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/full_data.csv', 'r', encoding='UTF8', newline = '')
lines = f.readlines()
with open('/home/arnavj/multimodal-learning/clip_train_eval/analysis/sorted_inds.pickle', 'wb') as f:

    pickle.dump(inds, f)

