This folder is to reproduce the results of continuous datasets. Each file is described as follows:

"Data Generation": In this file, we can generate continuous datasets.

"mixedcca" and "mixedcca-comptime": In these two files, we can run the MixedCCA algorithm and obtain its computation time. At the same time, we can generate the initial values that will be used in simulated tempering.

"ST_main": In this file, we can run our simulated tempering algorithm.


"rifle" and "rifle-time": In these two files, we can run the Rifle algorithm, and obtain its computation time. In those two files, We slightly modified the function rifle::rifle() in order to record the trace of the estimation (i.e., the value of _xprime_ at each iteration), so that we can produce the trace plot for Rifle algorithm in Figure 1 in our paper. All other arguments in the function rifle() remain the same.

"Comparison": In this file, we can compare the results for all algorithms in terms of error, TPR and TNR.
