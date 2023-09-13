from abc import ABC, abstractmethod
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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
import os
import math
import pandas as pd
df = pd.read_csv('/home/arnavj/multimodal-learning/clip_train_eval/dsets/' + "easy1M" + ".csv", sep=',')   
df2 = pd.read_csv('/home/arnavj/multimodal-learning/clip_train_eval/dsets/' + 'cross-ti-0.3-1M' + ".csv", sep=',')

df_merge = df2.merge(df,how='inner')
print(len(df_merge['caption']))
df3 = pd.read_csv("/home/arnavj/multimodal-learning/clip_train_eval/dsets/full_data.csv")
df3["index"] = df3.index

df_merge2 = df.merge(df3,how='inner')
np_arr = df_merge2['index'].to_numpy()
block_size = 512
similarity_matrices = []
X1 = torch.load('full_text_embeds2.pt', map_location='cuda')
X1 = X1[torch.from_numpy(np_arr)]

X2 = torch.load('full_image_embeds2.pt', map_location='cuda')
X2 = X2[torch.from_numpy(np_arr)]
print(len(X1))
print(len(X2))
sum = 0
for i in tqdm(range(X1.shape[0] // block_size + 1)):
    similarity_matrices_i = []
    e = X1[i*block_size:(i+1)*block_size]
    e_t = X2[i*block_size:(i+1)*block_size]
    '''
    similarity_matrices.append(
        np.array(
        torch.cosine_similarity(e,e_t).detach().cpu()
        )
    )'''
    sum += np.sum(np.array(torch.cosine_similarity(e,e_t).detach().cpu()))

print(sum/len(X1))
    
#similarity_matrix = np.block(similarity_matrices).astype(np.float16)
#sorted_matrix = np.sort(similarity_matrix)
'''
plt.plot(sorted_matrix.tolist())
plt.ylabel("Cos Similarity")
plt.title("Cosine Similarity Distribution")
plt.xlabel('Index')
#plt.xlim(1000,200000)
plt.savefig('cos_full2.png')

'''
#inds = similarity_matrix.argsort()
'''
f = open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/full_data.csv', 'r', encoding='UTF8', newline = '')
lines = f.readlines()
with open('/home/arnavj/multimodal-learning/clip_train_eval/analysis/sorted_inds.pickle', 'wb') as f:

    pickle.dump(inds, f)
    '''

