from abc import ABC, abstractmethod
import os

import numpy as np
import torch
import clip
import pandas as pd
from tqdm.notebook import tqdm
from pkg_resources import packaging
from PIL import Image 
from tqdm import tqdm 
import torch

class Subset(ABC):
    def __init__(self, data_path, cos_path, percent):
        self.data_path = data_path
        self.cos_path = cos_path
        self.percent = percent
    @abstractmethod
    def get_indices():
        pass

    def get_subset(self, data_path, percent):
        data = pd.read_csv(data_path)
        indices = Subset.get_indices(percent)
        
        pass
    

class RandomSubset(Subset):
    def __init__(self, data_path, cos_path, percent):
        super().__init__(data_path, cos_path, percent)
        
    def get_indices(self, partition, percent):
        return np.indices(len(partition), int(len(partition) * percent))

class CosineSubset(Subset):
    def __init__(self, data_path, cos_path, percent):
        super().__init__(data_path, cos_path, percent)
    def get_embeddings(self):
        pass
    
    @abstractmethod
    def get_partition_subset():
        pass

    def get_indices(filepath):
        indices = np.empty()
        for file in os.listdir("/mydir"):
            if file.endswith(".npy"):
                indices.append(CosineSubset.get_partition_subset())
        indices = np.unique(indices)
    

class EasySubset(CosineSubset):
    def __init__(self, data_path, cos_path, percent):
        super().__init__(data_path, cos_path, percent)
    def get_dir(self):
        return ["image_text"]
    def get_partition_subset(partition, percent):
        return np.argsort(partition.diagonal())[int(len(partition) * (1 - percent)):]
    pass
class HardSubset(CosineSubset):
    def get_dir(self):
        return ["image_text"]
    def get_partition_subset(partition, percent):
        return np.argsort(partition.diagonal())[0:int(len(partition) * percent)]
        

class ExperimentSubset(CosineSubset):
    def __init__(self, data_path, cos_path, percent):
        super().__init__(data_path, cos_path, percent)
    def get_indices(self, similarities, percent):
        final = np.empty()
        for i in similarities:
            num_elements = len(i) * percent
            for x in range(num_elements):
                sums = np.sum(i, axis=1)
                ind = np.argmax(sums)
                final.append(ind)
                sums[:,ind] = 0
                sums[ind,:] = 0

        

class Experiment1Subset(ExperimentSubset):
    def __init__(self, data_path, cos_path, percent):
        super().__init__(data_path, cos_path, percent)
    def get_dir(self):
        return ["image_text"]
    def get_similarity(self):
        pass

class Experiment2Subset(ExperimentSubset):
    def __init__(self, data_path, cos_path, percent):
        super().__init__(data_path, cos_path, percent)
    def get_dir(self):
        return ["image_image","text_text"]
    def get_similarity(self):
        pass

class Experiment3Subset(ExperimentSubset):
    def __init__(self, data_path, cos_path, percent):
        super().__init__(data_path, cos_path, percent)
    def get_dir(self):
        return ["image_image","text_text"]
    def get_similarity(self):
        
        pass