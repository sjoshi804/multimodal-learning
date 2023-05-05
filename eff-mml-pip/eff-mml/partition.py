from typing import List
import json 

def partition_by_caption(path_to_classes: str, captions: List[str]):
    with open(path_to_classes, "rb") as f:
        classes = json.load(f)

    partition = {}
    for latent_class in classes:
        partition[latent_class] = []

    def latent_class_in_caption(caption: str, latent_class: str):
        return latent_class in caption

    for i, caption in enumerate(captions):
        for latent_class in classes:
            if latent_class_in_caption(caption=caption, latent_class=latent_class):
                partition[latent_class].append(i)

