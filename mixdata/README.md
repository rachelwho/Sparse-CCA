This folder is to reproduce the results of mixed datasets. We still generate the continuous datasets first (by "Data Generation" file in continuous folder), and then truncated the data when we run the algorithm. Each file functions as follows:

"initial": In this file, we can get initial values for all algorithms.

"estimateR": In this file, we can get an estimator for the covariance matrices that can be used for all algorithms.

"mixedcca" and "mixedcca-computation": In this file, we can run MixedCCA algorithm and we can obtain the computation time for MixedCCA algorithm. At the same time, we can generate the initial values that can also be used in simulated tempering.

"mixed data types": In this file, we can run our simulated tempering algorithm.

"result": In this file, we can compare the results for all algorithms in terms of error, TPR and TNR.