#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:27:44 2021

@author: apple
"""
import sys
sys.path.append('..')
from SCCA import *
import pandas as pd
import pickle

Niter = int(sys.argv[1])
nrun = int(sys.argv[2])
color = ['blue', 'red']
label = [r'$n < s^2 log(p)$', 'n = p/2']
res = np.zeros((3,2,nrun,Niter))

i1 = int(0)
u = 1.5
s = 6
lam = 0.9
fig, ax = plt.subplots(3, 1, sharey=False, tight_layout=True , figsize = (10,10))
for d in [5000]:
    ax[i1].plot([0,Niter], [lam, lam])
    ax[i1].set_xlabel("Iteration")
    ax[i1].set_ylabel(r"rayleigh quotient")
    ax[i1].set_title(r"rayleigh quotient value when p=%i" %d)
    ax[i1].grid()
    i2 = 0
    for n in [int(s**u*np.log(d)), d//2]:
        
        
        
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
        Sigma = np.concatenate((np.concatenate((SigX, SigXY), axis=0),
                                np.concatenate((SigXY.T, SigY), axis=0)), axis = 1)

        for ii in range(nrun):
            # now1 = datetime.now()
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
            
            
            
            error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam, I, phi,aa, visits = Simu_t(Ahat, Bhat, vx, vy, d, n, temp = np.arange(1,0.6, -0.1)
                                                        , sigmasq = 1,initial=False, Niter = Niter, update = True, rho_0= 10, loggam = np.zeros(4) -4)
        #     burnin = 700
            # ll = Res_delta[I==0].shape[0]
            res[i1, i2, ii, :] = error[:, 1]

            
        #     error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam = ParaBayes(Ahat, Bhat, vx, vy, d, n, temp=[1]
        #                                                 , sigmasq = 1,initial=False, Niter = 2000, update = True)
            
        #     results[ii, 2] = np.mean(error[burnin:,2])
        #     results[ii, 3] = np.mean(error[burnin:,3])
        
    
        ax[i1].plot( np.mean(res[i1, i2, :, :], axis = 0), '-', color = color[i2],  alpha=0.5, label =  label[i2])
        
        
        
        #     ax[1].plot( error[I==0,3], 'b.', alpha=0.5 )
        #     ax[1].set_xlabel("Iteration")
        #     ax[1].set_ylabel(r"$error$")
        #     ax[1].set_title(r"error for $v_y$ when p=%i & N=%i" % (d, n))
        #     ax[1].grid()
        
        #     ax[2].plot(Res_theta[I==0, 0]) 
        #     ax[2].set_title(r'$\theta$')
        
        #     ## eigenvalue
        #     ax[3].plot(error[I==0,1], 'b.')
        #     ax[3].set_title(r"eigenvalues")
        #     ax[3].grid()
            
        #     plt.show()
            # print(datetime. now() - now1)
        i2 = i2 + 1
    ax[i1].legend()
    with open('samplesize-%i.pkl' %i1, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(res[i1,:,:,:], f)
    i1 = i1 + 1
    
plt.savefig('fig/samplesize.png')
plt.close()
with open('samplesize.pkl' , 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(res, f)
