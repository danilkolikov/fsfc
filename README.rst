================================
Feature Selection for Clustering
================================

|mit|

**FSFC** is a library with algorithms of feature selection for clustering.

It's based on the article `"Feature Selection for Clustering: A Review." <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.295.8115>`_
by S. Alelyani, J. Tang and H. Liu

Algorithms are covered with tests that check their correctness and compute some clustering metrics.
For testing we use open datasets:

- Generic data - `High-dimensional points datasets <http://cs.uef.fi/sipu/datasets/>`_
- Text data - `SMS Spam Collection <https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection>`_


Implemented algorithms:
-----------------------

- Generic Data:
    - SPEC family - NormalizedCut, ArbitraryClustering, FixedClustering
    - Sparse clustering - Lasso
    - Localised feature selection - LFSBSS algorithm
    - Multi-Cluster Feature Selection
    - Weighted K-means
- Text Data:
    - Text clustering - Chi-R algorithm, Feature Set-Based Clustering (FTC)
    - Frequent itemset extraction - Apriori

Dependencies:
-------------

- numpy
- scikit-learn
- scipy

How to use:
-----------
Now the project is in the early alpha stage, so it isn't publish to pip.

Because of it, installation of the project is a bit complicated. To use **FSFC** you should:

1. Clone repository to your computer.
2. Run ``make init`` to install dependencies.
3. Copy content of the folder **fsfc** to the source root of your project.

After it you can use feature selectors as follows:

.. code:: python

    import numpy as np
    from fsfc.generic import NormalizedCut
    from sklearn.pipeline import Pipeline
    from sklearn.cluster import KMeans

    data = np.array([...])

    pipeline = Pipeline([
        ('select', NormalizedCut(3)),
        ('cluster', KMeans())
    ])
    pipeline.fit_predict(data)

How to support:
---------------
You can support development by testing and reporting of bugs or opening pull-requests.

Project has tests, they can be run with the command ``make test``

Also code there is a Sphinx documentation for code, it can be built with the command ``make html``

References:
-----------

- Alelyani, Salem, Jiliang Tang, and Huan Liu. `"Feature Selection for Clustering: A Review." <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.295.8115>`_
    Data Clustering: Algorithms and Applications 29 (2013): 110-121.
- Zhao, Zheng, and Huan Liu. `"Spectral feature selection for supervised and unsupervised learning." <http://www.public.asu.edu/~huanliu/papers/icml07.pdf>`_
    Proceedings of the 24th international conference on Machine learning. ACM, 2007.
- D.M. Witten and R. Tibshirani. `"A framework for feature selection in clustering." <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2930825/>`_
    Journal of the American Statistical Association, 105(490):713–726, 2010.
- Li, Yuanhong, Ming Dong, and Jing Hua. `"Localized feature selection for clustering." <http://www.cs.wayne.edu/~jinghua/publication/PRL-LocalizedFeatureSelection.pdf>`_
    Pattern Recognition Letters 29.1 (2008): 10-18.
- Cai, Deng, Chiyuan Zhang, and Xiaofei He. `"Unsupervised feature selection for multi-cluster data." <https://dl.acm.org/citation.cfm?id=1835848>`_
    Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2010.
- Huang, Joshua Zhexue, et al. `"Automated variable weighting in k-means type clustering." <https://ieeexplore.ieee.org/document/1407871/>`_
    IEEE Transactions on Pattern Analysis and Machine Intelligence 27.5 (2005): 657-668.
- Li, Yanjun, Congnan Luo, and Soon M. Chung. `"Text clustering with feature selection by using statistical data." <https://ieeexplore.ieee.org/document/4408578/>`_
    IEEE Transactions on knowledge and Data Engineering 20.5 (2008): 641-652.
- Agrawal, Rakesh, and Ramakrishnan Srikant. `"Fast algorithms for mining association rules." <http://www.vldb.org/conf/1994/P487.PDF>`_
    Proc. 20th int. conf. very large data bases, VLDB. Vol. 1215. 1994.
- Beil, Florian, Martin Ester, and Xiaowei Xu. `"Frequent term-based text clustering" <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.12.7997&rep=rep1&type=pdf>`_
    Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2002.

.. |mit| image:: https://img.shields.io/github/license/mashape/apistatus.svg