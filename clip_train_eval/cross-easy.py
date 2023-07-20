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
import pickle
import math


percent = 0.287
percent2 = 0.287
f = open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/imagenet_like_train2.csv', 'r', encoding='UTF8', newline = '')
lines = f.readlines()
partition_file = open('analysis/partition_imagenet_like', 'rb')
partition = pickle.load(partition_file)
indices = {}
curr_best = {}
lens = {}
full = set()
indices2 = set()
for i in tqdm(range(0, 998)):
    indices[i] = []
    similarities2 = np.load(open('/home/arnavj/multimodal-learning/subset_creation/image_text-cos-sim/' + str(i) + '.npy', 'rb'))
    similarities2 = similarities2.astype(np.float32)
    #similarities = np.load(open('image_text-cos-sim/' + str(i) + '.npy', 'rb'))
    #similarities += np.load(open('text_text-cos-sim/' + str(i) + '.npy', 'rb'))
    #similarities = np.load(open('/home/arnavj/multimodal-learning/clip_train_eval/text_text-cos-sim/' + str(i) + '.npy', 'rb'))
    #similarities = np.maximum(similarities, np.load(open('/home/arnavj/multimodal-learning/clip_train_eval/image_image-cos-sim/' + str(i) + '.npy', 'rb')))
    #similarities += np.load(open('/home/arnavj/multimodal-learning/clip_train_eval/image_image-cos-sim/' + str(i) + '.npy', 'rb'))
    similarities = similarities2 + similarities2.transpose()
    similarities = similarities.astype(np.float32)
    lens[i] = len(similarities)
    num_elements = math.ceil(len(similarities) * percent)
    #num_elements = len(similarities)
    conversion = partition[list(partition.keys())[i]]
    if(len(similarities) != len(conversion )):
        print("error")
        print(str(i) + " " + str(len(similarities)) + " " + str(len(conversion)))
    sums = np.ma.array(np.sum(similarities, axis=1))
    temp = []
    for x in range(num_elements):
        if((x+1)%1000 == 0):
            print(str(i) + " " + str(x))
        ind = sums.argmax()
        
        sums[ind] = np.ma.masked
        #indices2.add(conversion[ind])
        #if conversion[ind] not in curr_best or curr_best[conversion[ind]][0] > x/num_elements:
        #    curr_best[conversion[ind]] = (x/num_elements, i)
        temp.append((similarities2[ind][ind],ind))
        #full.add(conversion[ind])
        sums = sums - similarities[:,ind]
    temp.sort(reverse=True)
    num_elements2 = math.ceil(len(temp) * percent)
    for x in range(num_elements2):
        _, temp2 = temp[x]
        full.add(conversion[temp2])

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
with open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/cross-easy.csv', 'w', encoding='UTF8', newline = '') as f:

    for i in full:
        f.write(lines[i + 1 ])  

