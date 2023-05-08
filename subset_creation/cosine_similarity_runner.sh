#!/bin/bash

# Define script arguments
partition=$1
x1=$2
x2=$3
latent_class_start_idx=$4
latent_class_end_idx=$5
device=$6
block_size=$7
run_prefix=$8

# Loop through latent class indices and call cosine_similarity.py
for (( i=$latent_class_start_idx; i<=$latent_class_end_idx; i++ ))
do
    python cosine_similarity.py \
        --partition $partition \
        --x1 $x1 \
        --x2 $x2 \
        --latent-class-idx $i \
        --device $device \
        --block-size $block_size \
        --run-prefix $run_prefix
done
