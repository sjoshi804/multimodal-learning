from tqdm import tqdm 
from typing import List, Optional, Callable
import json 

def partition_by_caption(classes: List[str], captions: List[str], partition_function: Optional[Callable] = None, verbose: bool = False):
    
    partition = {}
    for latent_class in classes:
        partition[latent_class] = []

    def latent_class_in_caption(caption: str, latent_class: str):
        latent_classes = [latent_class]
        if "/" in latent_class:
            latent_classes = latent_class.split("/")
            latent_classes = [word.strip() for word in latent_classes]
        return any([lc in caption for lc in latent_classes])
    
    if partition_function is None:
        partition_function = latent_class_in_caption

    for i, caption in tqdm(enumerate(captions), total=len(captions), disable=not verbose):
        for latent_class in classes:
            if partition_function(caption=caption, latent_class=latent_class):
                partition[latent_class].append(i)

    return partition
