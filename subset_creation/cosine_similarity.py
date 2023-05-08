import argparse
from datetime import datetime
import pickle 
from eff_mml.util import compute_cosine_similarity
import torch 
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Cosine Similairty by Partition')
    parser.add_argument("--partition", type=str, help="Path to partition (dict object)")
    parser.add_argument("--x1", type=str, help="Path to tensors for X1")
    parser.add_argument("--x2", type=str, default="Path to tensors for X2")
    parser.add_argument("--latent-class-idx", type=int, default="index of latent class to run for")
    parser.add_argument("--device", type=int, default="GPU ID")
    parser.add_argument("--block-size", type=int, default="size of block for cosine sim calculation")
    parser.add_argument("--run-prefix", type=str)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    # Load 
    with open(args.partition, "rb") as f:
        partition = pickle.load(f)
    X1 = torch.load(args.x1, map_location=device)
    X2 = torch.load(args.x2, map_location=device)
    latent_classes = list(partition.keys())

    save_dir = f"{args.run_prefix}-cos-sim"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    compute_cosine_similarity(
        X1=X1[partition[latent_classes[args.latent_class_idx]]],
        X2=X2[partition[latent_classes[args.latent_class_idx]]],
        block_size=512,
        filename=f"{save_dir}/{args.latent_class_idx}",
    )

