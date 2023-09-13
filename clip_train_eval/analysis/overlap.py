import os
import math


def getoverlap(file1, file2):
    f2 = open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/' + file1 + ".csv", 'r', encoding='UTF8', newline = '')
    f3 = open('/home/arnavj/multimodal-learning/clip_train_eval/dsets/' + file2 + ".csv", 'r', encoding='UTF8', newline = '')
    lines = f2.readlines()
    lines2 = f3.readlines()
    print(len(set(lines).intersection(lines2)))
    return len(set(lines).intersection(lines2))/min(len(lines),len(lines2))

#dsets = ["imagenet_like_hard12","imagenet_like_easy12", "cross-sum","intra-sum", "intra-max","semdedup12", "cross-sum2", "intra-sum2","intra-max2", "cross-sum-best"]
dsets = ["easy_20", "easy20_rand20_model", "easy20_full_model"]
for i in dsets:
    for x in dsets:
        print(i + "    " + x + "     " + str(getoverlap(i,x)))