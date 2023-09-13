# ESCAPED

This repo contains the source of code of our paper, [ESCAPED: Efficient Secure and Private Dot Product Framework for Kernel-based Machine Learning Algorithms with Applications in Healthcare](https://ojs.aaai.org/index.php/AAAI/article/view/17199), published in AAAI 2021.

## How to replicate the experiments
In order to run the experiments that we conducted to analyze our solution in the paper, we provide three scripts:
1. **_classification\_exp\_runner.sh <num\_of\_input-parties>_**: To see the scalability of our solution to the number of parties up to 6 parties, you can run this script by providing the number of parties as an argument. The results will be saved under "results/hiv_coreceptor_prediction/num_of_input-parties/<number_of_input-parties>_parties/run<run_number>/".
2. **_classification\_dataset\_size\_exp\_runner.sh <dataset\_size>_**: To analyze how scalable ESCAPED is to varying the size of the dataset, you can run this script by providing the dataset size as _full_, _half_ or _quarter_. The results will be saved under "results/hiv_coreceptor_prediction/dataset_size/<dataset_size>/run<run_number>/".
3. **_clustering\_exp\_runner.sh_**: In order to make multi-omics dimensionality reduction and clustering on HNSC patients, you can run this script that conducts the experiments and saves the results in "results/clustering/" and the kernel matrices in "kms/". You can, then, use these kernel matrices to run the [web-rMKL](https://academic.oup.com/nar/article/47/W1/W605/5494746) on [its website](https://web-rmkl.org/home/upload/) to obtain the clustering results. Furthermore, if you just want to run the web-rMKL via the kernel matrices which will be computed by ESCAPED, we already have them in "kms/". Note that the required data for the clustering task is zipped and you need to unzip them before running the corresponding script. You can find them under "data/cancer_dataset/hnsc/".

# Randomized Encoding Based Approach
In order to run the randomized encoding based approach via three input-parties for varying dataset size, run randomized_encoding_approach/classification_with_re_exp_runner <dataset_size>. Since we did not further evaluate the randomized encoding based approach in the varying number of input-parties, we fixed the number of input-parties to 3. The results will be saved in "randomized_encoding_approach/results/".
