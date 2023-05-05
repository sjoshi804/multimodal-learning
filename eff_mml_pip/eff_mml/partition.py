from tqdm import tqdm 
from typing import List
import json 

def partition_by_caption(classes: List[str], captions: List[str], verbose: bool = False):
    partition = {}
    for latent_class in classes:
        partition[latent_class] = []

    def latent_class_in_caption(caption: str, latent_class: str):
        latent_classes = [latent_class]
        if "/" in latent_class:
            latent_classes = latent_class.split("/")
            latent_classes = [word.strip() for word in latent_classes]
        return any([latent_class in caption for latent_class in latent_classes])

    for i, caption in tqdm(enumerate(captions), total=len(captions), disable=not verbose):
        for latent_class in classes:
            if latent_class_in_caption(caption=caption, latent_class=latent_class):
                partition[latent_class].append(i)

    return partition

