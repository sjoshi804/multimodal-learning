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

partition_file = open('analysis/partition_cluster', 'rb')
partition = pickle.load(partition_file)
indices = {}
curr_best = {}
lens = {}
full = set()
indices2 = set()
partition2 = {}
for i in tqdm(range(0, 1000)):
    indices[i] = []
    similarities = np.load(open('/home/arnavj/multimodal-learning/clip_train_eval/image-text-full-cos-sim/' + str(i) + '.npy', 'rb'))
    
    similarities = similarities.astype(np.float32)
    lens[i] = len(similarities)
    sorting = np.flip(np.argsort(np.diagonal(similarities)))
    #num_elements = len(similarities)
    conversion = np.array(partition[list(partition.keys())[i]])
    if(len(similarities) != len(conversion )):
        print("error")
        print(str(i) + " " + str(len(similarities)) + " " + str(len(conversion)))
    partition2[list(partition.keys())[i]] = conversion[sorting]

partition_file2 = open('analysis/partition_cluster_sort', 'wb')
partition = pickle.dump(partition2, partition_file2)


