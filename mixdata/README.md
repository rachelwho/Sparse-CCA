This folder is to reproduce the results of the mixed datasets. We still generate the continuous datasets first (by the "Data Generation" file in the "continuous" folder), and then truncate the data when we run the algorithm. Each file is described as follows:

"initial": In this file, we can get initial values for all algorithms.

"estimateR": In this file, we can get an estimator for the covariance matrices that will be used for all algorithms.

"mixedcca" and "mixedcca-computation": In these two files, we can run the MixedCCA algorithm and obtain its computation time. At the same time, we can generate the initial values that will be used in simulated tempering.

"mixed data types": In this file, we can run our simulated tempering algorithm.

"result": In this file, we can compare the results for all algorithms in terms of error, TPR and TNR.