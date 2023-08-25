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
import math
partition_file = open('analysis/predict_zero_sort', 'rb')
partition = pickle.load(partition_file)
file2 = open('full_trained_sort.pickle', 'rb')
file2_load = pickle.load(file2)
print(len(file2_load))
curriculum = {}

method = 1
if(method == 0):
    
    full = set()
    for i in tqdm(range(1, 31)):
        inds = []
        for x in (range(0, 1000)):
            vals = list(partition[list(partition.keys())[x]])
            length = math.ceil(len(vals) * 1000000/2837815)
            offset = math.ceil((len(vals) - length) * (i - 1) / 29)
            inds.extend(vals[offset : offset + length])
            full.update(vals[offset : offset + length])
            
            
        curriculum[i] = inds
    

else:
    full = set()
    sets2 = set()
    lens = {}
    lists = {}
    for x in (range(0, 1000)):
        lists[x] = list(partition[list(partition.keys())[x]])
        lens[x] = 0
    for i in tqdm(range(1, 31)):
        inds = []
        curr = 0
        if(i == 1):
            while(len(inds) < 1000000):
                for x in (range(0, 1000)):
                    vals = lists[list(partition.keys())[x]]
                    if(curr >= len(vals)):
                        continue
                    inds.append(vals[curr])
                    full.add(vals[curr])
                    lens[list(partition.keys())[x]] = curr + 1
                    print(len(inds), end = '\r')
                curr += 1
        else:
            for x in (range(0, 1000)):
                vals = lists[list(partition.keys())[x]]
                length = lens[list(partition.keys())[x]]
                
                offset = math.ceil((len(vals) - length) * (i - 1) / 29)
                inds.extend(vals[offset : offset + length])
                full.update(vals[offset : offset + length])
                
        curriculum[i] = inds
        

print(len(full))
curr_file = open('curriculum_rr_zero_full.pickle', 'wb')
pickle.dump(curriculum, curr_file)