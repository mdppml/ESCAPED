# ESCAPED

This repo contains the source of code of our paper, [ESCAPED: Efficient Secure and Private Dot Product Framework for Kernel-based Machine Learning Algorithms with Applications in Healthcare](https://ojs.aaai.org/index.php/AAAI/article/view/17199), published in AAAI 2021.

## How to replicate the experiments
In order to run the experiments that we conducted in our analysis, we provide three scripts:
1. **_classification\_exp\_runner.sh_**: This can be run to see the scalability of our solution to the number of parties up to 6 parties. You need to provide <num_of_input-parties> to the script as an argument.
2. **_classification\_dataset\_size\_exp\_runner.sh_**: In order to analyze how scalable ESCAPED is to varying the size of the dataset, you can run this script by providing <dataset_size> as "full", "half" or "quarter".
3. **_clustering\_exp\_runner.sh_**: In order to make multi-omics dimensionality reduction and clustering on HNSC patients, you can run this script that conducts the experiments and saves the results in "results/clustering/" and the kernel matrices in "kms/". You can, then, use these kernel matrices to run the [web-rMKL](https://academic.oup.com/nar/article/47/W1/W605/5494746) to obtain the clustering results. Furthermore, if you just want to run the web-rMKL via the kernel matrices which will be computed by ESCAPED, we already have them in "kms/".
