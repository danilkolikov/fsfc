================================
Feature Selection for Clustering
================================

A library with algorithms for feature selection for clustering.

Based on the paper "Feature Selection for Clustering" by S. Alelyani, J. Tang and H. Liu

Uses high-dimensional points datasets downloaded from http://cs.uef.fi/sipu/datasets/
and text dataset https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection for testing

Implemented algorithms:

- SPEC family - NormalizedCut, ArbitraryClustering, FixedClustering
- Sparse clustering - Lasso
- Localised feature selection - LFSBSS algorithm
- Multi-Cluster Feature Selection
- Weighted K-means
- Text clustering - Chi-R algorithm, Feature Set-Based Clustering (FTC)
- Frequent itemset extraction - Apriori

Reference:

- Alelyani, Salem, Jiliang Tang, and Huan Liu. "Feature Selection for Clustering: A Review."
    Data Clustering: Algorithms and Applications 29 (2013): 110-121.
- Zhao, Zheng, and Huan Liu. "Spectral feature selection for supervised and unsupervised learning."
    Proceedings of the 24th international conference on Machine learning. ACM, 2007.
- D.M. Witten and R. Tibshirani. "A framework for feature selection in clustering."
    Journal of the American Statistical Association, 105(490):713–726, 2010.
- Li, Yuanhong, Ming Dong, and Jing Hua. "Localized feature selection for clustering."
    Pattern Recognition Letters 29.1 (2008): 10-18.
- Cai, Deng, Chiyuan Zhang, and Xiaofei He. "Unsupervised feature selection for multi-cluster data."
    Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2010.
- Huang, Joshua Zhexue, et al. "Automated variable weighting in k-means type clustering."
    IEEE Transactions on Pattern Analysis and Machine Intelligence 27.5 (2005): 657-668.
- Li, Yanjun, Congnan Luo, and Soon M. Chung. "Text clustering with feature selection by using statistical data."
    IEEE Transactions on knowledge and Data Engineering 20.5 (2008): 641-652.
- Agrawal, Rakesh, and Ramakrishnan Srikant. "Fast algorithms for mining association rules."
    Proc. 20th int. conf. very large data bases, VLDB. Vol. 1215. 1994.
Beil, Florian, Martin Ester, and Xiaowei Xu. "Frequent term-based text clustering."
    Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2002.