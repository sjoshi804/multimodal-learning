from abc import ABC, abstractmethod
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import numpy as np
import torch
import clip
import pandas as pd
from tqdm.notebook import tqdm
from pkg_resources import packaging
from PIL import Image 
from tqdm import tqdm 
import torch
from random import shuffle
import pickle
percent = 0.3526
f = open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/full_data.csv', 'r', encoding='UTF8', newline = '')
lines = f.readlines()
partition_file = open('analysis/partition_cluster', 'rb')
partition = pickle.load(partition_file)
indices = {}
curr_best = {}
lens = {}
full = set()
indices2 = set()
for i in tqdm(range(0, 998)):
    indices[i] = []
    similarities = np.load(open('/home/arnavj/multimodal-learning/clip_train_eval/image-text-full-cos-sim/' + str(i) + '.npy', 'rb'))
    #similarities = np.load(open('image_text-cos-sim/' + str(i) + '.npy', 'rb'))
    #similarities += np.load(open('text_text-cos-sim/' + str(i) + '.npy', 'rb'))
    #similarities = np.load(open('/home/arnavj/multimodal-learning/clip_train_eval/text_text-cos-sim/' + str(i) + '.npy', 'rb'))
    #similarities = np.maximum(similarities, np.load(open('/home/arnavj/multimodal-learning/clip_train_eval/image_image-cos-sim/' + str(i) + '.npy', 'rb')))
    #similarities += np.load(open('/home/arnavj/multimodal-learning/clip_train_eval/image_image-cos-sim/' + str(i) + '.npy', 'rb'))
    similarities = similarities + similarities.transpose()
    similarities = similarities.astype(np.float32)
    lens[i] = len(similarities)
    num_elements = int(len(similarities) * percent)
    #num_elements = len(similarities)
    conversion = partition[list(partition.keys())[i]]
    if(len(similarities) != len(conversion )):
        print("error")
        print(str(i) + " " + str(len(similarities)) + " " + str(len(conversion)))
    sums = np.ma.array(np.sum(similarities, axis=1))
    
    for x in range(num_elements):
        ind = sums.argmax()
        
        sums[ind] = np.ma.masked
        #indices2.add(conversion[ind])
        #if conversion[ind] not in curr_best or curr_best[conversion[ind]][0] > x/num_elements:
        #    curr_best[conversion[ind]] = (x/num_elements, i)
        full.add(conversion[ind])
        sums = sums - similarities[:,ind]
'''
print(len(curr_best))
print(len(indices2))
for i in curr_best:
    indices[curr_best[i][1]].append(i)
for i in range(0, 998):
    for x in range(0, int(len(indices[i]) * percent)):
        if(x >= len(indices[i])):
            print("error")
            break
        full.append(indices[i][x])
print(len(full))
'''
print(len(full))
full = list(full)
shuffle(full)
with open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/cross1M.csv', 'w', encoding='UTF8', newline = '') as f:
    f.write("caption,file\n")
    for i in range(0,min(len(full),1000000)):
        f.write(lines[full[i] + 1 ])  

