Methods for text data
=====================

FSFC contains implementation of some feature selection algorithms working with text data.
Main difference of such algorithms is that they accept SciPy Sparse Matrices as input. They can be
computed using vectorizers from :mod:`sklearn`

Every algorithm can be imported either from it's package or from the :mod:`fsfc.text` module

Chi-R algorithm
---------------
.. automodule:: fsfc.text.CHIR

Frequent Term-based Clustering
------------------------------
.. automodule:: fsfc.text.FTC
