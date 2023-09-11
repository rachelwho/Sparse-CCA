# Sparse-CCA

This repository contains the python implementation and appendices of the [Minimax Quasi-Bayesian estimation in sparse canonical correlation analysis via a Rayleigh quotient function](https://arxiv.org/pdf/2010.08627.pdf) by Qiuyun Zhu and Yves Atchade.

Please cite our paper if you find this code useful in your research. 

## Introduction
Canonical correlation analysis (CCA) is a popular statistical technique for exploring the relationship between datasets. The estimation of sparse canonical correlation vectors has emerged in recent years as an important but challenging variation of the CCA problem, with widespread applications. Currently available rate-optimal estimators for sparse canonical correlation vectors are expensive to compute. We propose a quasi-Bayesian estimation procedure that achieves the minimax estimation rate, and yet is easy to compute by Markov Chain Monte Carlo (MCMC). The method builds on Rayleigh quotient and uses a re-scaled Rayleigh quotient function as a quasi-log-likelihood. We adopt a Bayesian framework that combines this quasi-log-likelihood with a spike-and-slab prior that serves to regularize the inference and promote sparsity. 

## Instruction
To reproduce the numerical results in the paper, you may find codes and instruction in each folder. Table 1 and Figure 1 are produced by the files in the folder /continuous. Figure 2, Figure 3, Table 2 and Table 3 are produced by the files in the folder /mixdata. Figure 4 and Figure 5 are produced by the files in the folder /covid. There are also results contained in the supplementary materials. Specifically, the results for estimating the mixing time of our algorithm are produced by the files in the folder /mixingtime, and the results for estimating the sample size condition for our algorithm are produced by the files in the folder /sample_size.

## Example
```
from SCCA import *
import pandas as pd
n = 200
d = int(500)
s = 6
lam = 0.9

np.random.seed(100)
## generate Sigma_x
SigX = np.zeros(shape = (d//2, d//2))
SigY = np.zeros(shape = (d//2, d//2))
block = np.empty(shape = (d//10, d//10))
vx = np.zeros(shape= (d//2, 1))
vy = np.zeros(shape= (d//2, 1))
for i in range(d//10):
    for j in range(d//10):
        block[i, j] = 0.8 ** abs(i - j)
        
for i in range(5):
    SigX[d//10*i : d//10*(i+1) ,d//10*i : d//10*(i+1) ] = block
    SigY[d//10*i : d//10*(i+1) ,d//10*i : d//10*(i+1) ] = block

for i in [0, 5, 10]:
    vx[i] = 1/np.sqrt(3)

vy = vx.copy()
vx = vx / np.sqrt(vx.T.dot(SigX).dot(vx))
vy = vy / np.sqrt(vy.T.dot(SigY).dot(vy))

SigXY = lam * SigX.dot(vx).dot(vy.T).dot(SigY)
Sigma = np.concatenate((np.concatenate((SigX, SigXY), axis=0),np.concatenate((SigXY.T, SigY), axis=0)), axis = 1)

# Generate data
Z =  stats.multivariate_normal.rvs(np.zeros(d), Sigma, n)
X = Z[:, 0:d//2]
Y = Z[:,d//2:]
# center columns of X

Sighat_X = (X-np.mean(X)).T.dot(X-np.mean(X))
Sighat_Y = (Y-np.mean(Y)).T.dot(Y - np.mean(Y))
Sighat_XY = (X-np.mean(X)).T.dot(Y - np.mean(Y))

Ahat = np.vstack((np.hstack((np.zeros(shape = (d//2, d//2)), Sighat_XY))
                    ,np.hstack((Sighat_XY.T, np.zeros(shape = (d//2, d//2))))))

Bhat = np.concatenate((np.concatenate((Sighat_X, np.zeros(shape = (d//2, d//2))), axis=0),
                    np.concatenate((np.zeros(shape = (d//2, d//2)), Sighat_Y), axis=0)), axis = 1) / (n*2)
 
error1, Res_theta1, Res_delta1, swap_atem1, swap1, mAcc1, loggam1, I1, phi1,aa1, visits1 = Simu_t(Ahat, Bhat, vx, vy, d, n, temp=np.arange(1,0.5, -0.1)
                                                , sigmasq = 1,initial=[Res_theta[0],Res_delta[0]], Niter = 4000, update = True, loggam=np.zeros(5) -4)

fig, ax = plt.subplots(4, 1, sharey=False, tight_layout=True , figsize = (10,10))

ax[0].plot( error1[I1==0, 2], 'b.', alpha=0.5 )
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel(r"$error$")
ax[0].set_title(r"error for $v_x$ when p=%i & N=%i" % (d, n))
ax[0].grid()

ax[1].plot( error1[I1==0,3], 'b.', alpha=0.5 )
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel(r"$error$")
ax[1].set_title(r"error for $v_y$ when p=%i & N=%i" % (d, n))
ax[1].grid()

ax[2].plot(Res_theta1[I1==0, 0]) 
ax[2].set_title(r'$\theta$')

## eigenvalue
ax[3].plot(error1[I1==0,1], 'b.')
ax[3].set_title(r"eigenvalues")
ax[3].grid()
plt.show()
```
