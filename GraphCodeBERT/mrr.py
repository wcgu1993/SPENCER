# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import os
import numpy as np
from more_itertools import chunked
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_batch_size', type=int, default=1000)
    args = parser.parse_args()
    languages = ['java']
    MRR_dict = {}
    for language in languages:
        file_dir = './new_test_results/{}'.format(language)
        accs1, accs3, accs5 = [], [], []
        ranks = []
        num_batch = 0
        for file in sorted(os.listdir(file_dir)):
            print(os.path.join(file_dir, file))
            with open(os.path.join(file_dir, file), encoding='utf-8') as f:
                batched_data = chunked(f.readlines(), args.test_batch_size)
                for batch_idx, batch_data in enumerate(batched_data):
                    num_batch += 1
                    correct_score = float(batch_data[batch_idx].strip().split('<CODESPLIT>')[-1])
                    scores = np.array([float(data.strip().split('<CODESPLIT>')[-1]) for data in batch_data])
                    rank = np.sum(scores >= correct_score)
                    
                    if rank <= 5:
                        accs5.append(1)
                    else:
                        accs5.append(0)
                    
                    if rank <= 3:
                        accs3.append(1)
                    else: 
                        accs3.append(0)
                    
                    if rank == 1:
                        accs1.append(1)
                    else:
                        accs1.append(0)
                        
                    ranks.append(rank)
        mean_acc1 = np.mean(np.array(accs1))
        mean_acc3 = np.mean(np.array(accs3))
        mean_acc5 = np.mean(np.array(accs5))
        mean_mrr = np.mean(1.0 / np.array(ranks))
        print("{} acc1: {}".format(language, mean_acc1))
        print("{} acc3: {}".format(language, mean_acc3))
        print("{} acc5: {}".format(language, mean_acc5))
        print("{} mrr: {}".format(language, mean_mrr))
        MRR_dict[language] = mean_mrr
    # for key, val in MRR_dict.items():
    #     print("{} mrr: {}".format(key, val))


if __name__ == "__main__":
    main()