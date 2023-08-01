# from easyspc import SPC

# import os
# import sys
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import random

# from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.stats as stats
import scipy.special as sp
# from functools import partial
# from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LassoCV
# from sklearn.metrics import f1_score
# from sklearn.model_selection import KFold, train_test_split
from numpy.linalg import norm
# import time
# from sklearn.cross_decomposition import CCA
from datetime import datetime


def gradient_computer(theta, A, B, sigmasq, rho_1):
    return (2*theta.dot(B).dot(theta)* A.dot(theta) 
                                     - 2*theta.dot(A).dot(theta)*B.dot(theta) ) / sigmasq / \
                                    (theta.dot(B).dot(theta))**2-theta * rho_1

def RQ(theta_init, A, B, Niter1,p):
    for jj in range(Niter1):
        Rth = theta_init.T.dot(A).dot(theta_init)
        G = 2 * (theta_init.T.dot(B).dot(theta_init)*A.dot(theta_init) - Rth*theta_init) / (theta_init.T.dot(B).dot(theta_init))**2
        theta_init = theta_init + 0.01*G
    
    thetahat = theta_init / np.sqrt((theta_init.T.dot(B).dot(theta_init)))
    delta = (np.abs(thetahat > 0.1) + 0).reshape(p)
    return thetahat, delta


def RW_Update_t(A, B, sigmasq, gam, rho_1, theta_in, t=1):
    theta = theta_in.reshape(len(theta_in), 1)
    thetahat_prop = theta + np.sqrt(gam) * np.random.normal(size=(len(theta_in), 1))
    RQ = 1 / sigmasq * theta.T.dot(A).dot(theta)/ theta.dot(B).dot(theta)* t
    RQ_prop = 1/ sigmasq * thetahat_prop.T.dot(A).dot(thetahat_prop)/ thetahat_prop.dot(B).dot(thetahat_prop) * t
    Acc = RQ_prop - RQ - (rho_1 /2) * (norm(thetahat_prop)**2 - norm(theta_in)**2) * t
    
    Acc = min(1, np.exp(Acc))
    if np.random.uniform(size=1) <= Acc:
        theta = thetahat_prop.copy()
        RQ = RQ_prop.copy()
    return theta[:,0], RQ, Acc

def MALA_t(A, B, sigmasq, tau,  rho_1, theta_in, t=1, truncate = False):
    theta = theta_in.copy()


    pp = len(theta)
    gradient = gradient_computer(theta, A, B, sigmasq, rho_1) * t
#     print(gradient)
    if truncate:
        gradient = np.sign(gradient) * np.where(np.abs(gradient)<truncate, np.abs(gradient), truncate)
    thetahat_prop = theta + tau* gradient + np.sqrt(2* tau ) * stats.multivariate_normal.rvs(0, 1, pp)
    RQ = 1 / sigmasq * theta_in.T.dot(A).dot(theta_in)/ theta.dot(B).dot(theta) *t
    RQ_prop = 1/ sigmasq * thetahat_prop.T.dot(A).dot(thetahat_prop)/ thetahat_prop.dot(B).dot(thetahat_prop) *t
    dif = thetahat_prop - theta
    Acc = RQ_prop - RQ - (rho_1 /2) * (norm(thetahat_prop)**2 - norm(theta)**2) * t -  (np.sum(
        (-dif - tau * gradient_computer(thetahat_prop, A, B, sigmasq, rho_1) * t)**2)- np.sum((dif - tau * gradient)**2))/4/tau
    Acc = min(1, np.exp(Acc))
    # print(gradient, Acc)
    if np.random.uniform(size=1) <= Acc:
        theta = thetahat_prop.copy()
        RQ = RQ_prop
    return theta, RQ, Acc

def loglik(A, B, sigmasq,  rho_0, rho_1, theta, delta, q):
    theta_d = theta*delta
    return 1 / sigmasq * (theta_d).T.dot(A).dot(theta_d)/ theta_d.dot(B).dot(theta_d) + sum(delta) * (0.5 * np.log(rho_1 /rho_0) 
                        + np.log(q) - np.log(1-q))- 0.5 * norm(theta*(1-delta))**2 *rho_0- 0.5 * norm(theta_d)**2 *(rho_1)

def Rifle(A, B, vx, vy, p, n,k,eta=0.01, initial = True, Niter = 5000):
    vx_new = vx/ norm(vx)
    vy_new = vy/ norm(vy)
    Res_theta = np.empty(shape = (Niter, p))
    Res_theta[0,:] = np.random.normal(size = p)
    if initial == True:
        v = np.concatenate([vx_new, vy_new]).reshape(p) + np.random.normal(size = p)*0.2
    else:
        v = Res_theta[0,:]
    ind = np.abs(v).argsort()[:(p-k)]
    v[ind] = 0
    v = v/norm(v)
    error = np.empty(shape = (Niter, 4))
    Res_theta[0,:] = v
    for i in range(Niter):
        error[i,1] = v.dot(A).dot(v) / v.dot(B).dot(v)
        C = np.identity(p) + eta / error[i,1] * (A - error[i,1] * B)
        v_new = C.dot(v)/ norm(C.dot(v))
        ind = np.abs(v_new).argsort()[:(p-k)]
        v_new[ind] = 0
    #         print(v_new)
        v = v_new
        v = v / norm(v)
        error[i, 2] = min((norm(v[:p//2] - vx_new[:,0])**2), (norm(v[:p//2] + vx_new[:,0])**2))
        error[i, 3] = min((norm(v[p//2:] - vy_new[:,0])**2), (norm(v[p//2:] + vy_new[:,0])**2))
        Res_theta[i,:] = v
    error[:,1] = error[:,1]/2/n
    return error, Res_theta
        

# def loglik(A, B, sigmasq,  rho_0, rho_1, theta, delta, q):
#     theta_d = theta*delta
#     return 1 / sigmasq * (theta_d).T.dot(A).dot(theta_d)/ theta_d.dot(B).dot(theta_d) + sum(delta * (0.5 * np.log(rho_1 /2 /np.pi) 
#                                                                     - 0.5 * theta**2 *rho_1 + np.log(q)) 
#                                                                        + (1-delta)* (0.5 * np.log(rho_0 /2 /np.pi)
#                                                                     - 0.5 * theta**2 *rho_0) + np.log(1-q))


def ParaBayes(A, B, vx, vy, p, n, temp=[1], sigmasq=1, initial = True, Niter = 3000, update = True, loggam = [-13]):
#     np.random.seed(100)
    vx = vx / norm(vx)
    vy = vy / norm(vy)
    ndim = len(temp)
    theta_init = np.random.normal(size = (ndim,p))
    
    

#     sigmasq = 0.5 / vartheta
#     sigmasq = 1 / np.linalg.eig(Sighat)[0][1]
    rho_1 = 0.5
    rho_0 = 10 ##n
    q = 1/ p**1.5

    
#     Niter = 4000
    
    loggam = np.zeros( ndim) -5
    error = np.empty(shape = (Niter, 4))
    Res_theta = np.empty(shape = (Niter, p))
    Res_delta = np.empty(shape = (Niter, p))
    mAcc = np.zeros(ndim)
    swap = np.zeros(ndim-1)
    swap_atem = np.zeros(ndim-1)
    theta0 = np.empty(shape = (ndim, p))
    delta0 = np.ones(shape = (ndim, p))
    t = temp
    theta0 = theta_init
#         t[i] = 1 / temp**i
#         t[i] = 1 / (1+i*temp)
        
    for jj in range(Niter):
        ## Update each coordinate
        for ii in range(ndim):
            
            ## Update delta
            for j in range(p):
                delta_j = delta0[ii,:] + 0
                delta_j[j] = 0
                theta_j = theta0[ii, delta_j == 1]
                if np.sum(theta_j!=0) == 0:
                    r0 = -float('inf')
    #                 print('NA1')
                else:
                    Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
                    B_j =  B[delta_j ==1, :][:, delta_j ==1]
                    r0 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
                delta_j[j] = 1
                theta_j = theta0[ii, delta_j == 1]
                Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
                B_j =  B[delta_j ==1, :][:, delta_j ==1]
                r1 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
                r = (r0 - r1 + np.log(1 - q) -  np.log(q) + np.log(rho_0 / rho_1) / 2 +\
                theta0[ii,j]**2 / 2 * (rho_1 - rho_0)) 
                prob = 1 / (1 + np.exp(r * t[ii]))
                
                delta0[ii,j]  = np.random.uniform(size = 1) <= prob
            ## Update theta

            theta0[ii, delta0[ii,:] == 0] = stats.multivariate_normal.rvs(0, 1/rho_0/t[ii], sum(delta0[ii,:] == 0))
            ind = np.where(delta0[ii,:] != 0)[0]
#             print(ind)
            if len(ind)==0:
                Acc = mAcc[ii]
            else:
                subSighat = A[ind,:][:, ind]
                subB = B[ind,:][:, ind]
                theta_in = theta0[ii, ind]
                theta_out, RQ, Acc = MALA_t(subSighat, subB, sigmasq, np.exp(loggam[ii]) , rho_1, theta_in, t[ii])
                
                theta0[ii, ind] = theta_out.copy()
            if update == True:
                loggam[ii] = loggam[ii] + 1/ (jj**0.6 + 1) * (Acc - 0.3)
            
            mAcc[ii] = mAcc[ii] + 1 / (jj + 1) *(Acc - mAcc[ii])
        if(ndim > 1):
            select = random.sample(list(np.arange(ndim-1)),1)[0]
#             print(loglik(Sighat, sigmasq, rho_0, rho_1, theta0[select,:], delta0[select,], q) )
            alpha = min(1, np.exp( (t[select + 1]- t[select]) * (loglik(A,B, sigmasq, 
                                                                            rho_0, rho_1, theta0[select,:], delta0[select,], q) -
                                                                     loglik(A, B, sigmasq, 
                                                                           rho_0, rho_1, theta0[select+1,:], delta0[select+1,], q)) ))
            
            swap_atem[select] = swap_atem[select] + 1
            if np.random.uniform(size=1) <= alpha:
                swap[select] = swap[select] + 1
                aa = theta0[select+1,:].copy()
                bb = delta0[select+1,:].copy()
                theta0[select+1,:] = theta0[select,:]
                theta0[select,:] = aa.copy()
                delta0[select+1,:] = delta0[select,:]
                delta0[select,:] = bb.copy()
            
            
        thetaout =  theta0[0,:]*(delta0[0,:]) 
        normcons = np.sqrt(thetaout.dot(B).dot(thetaout))
        thetaout = thetaout /  normcons
#         error[jj, 0] = norm(np.outer(thetaout, thetaout)  - proj_star1, 'fro')
#         print(thetaout.T.dot(A).dot(thetaout) / n / 2 / thetaout.dot(B).dot(thetaout))
#         print(thetaout[:p//2].dot(B[:p//2,:p//2]).dot(thetaout[:p//2]))
        error[jj, 1] = thetaout.T.dot(A).dot(thetaout) / n / 2
        norm1 = norm(thetaout[:p//2])
        norm2 = norm(thetaout[p//2:])
        error[jj, 2] = min((norm(thetaout[:p//2] / norm1 - vx[:,0])**2), (norm(thetaout[:p//2]/ norm1 + vx[:,0])**2))
        error[jj, 3] = min((norm(thetaout[p//2:] / norm2 - vy[:,0])**2), (norm(thetaout[p//2:] / norm2+ vy[:,0])**2))
#         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
        Res_theta[jj, :] = theta0[0, :] / normcons
        Res_delta[jj, :] = delta0[0, :]
#         print((Res_theta[jj,:]*Res_delta[jj,:]).dot(B).dot(Res_theta[jj, :]*Res_delta[jj,:]))

    return error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam



def mcmc(theta0, delta0,A, B, vx, vy, p, n, temp=[1], sigmasq=1, initial = True, Niter = 3000, update = True, loggam = [-5]):
    vx = vx / norm(vx)
    vy = vy / norm(vy)
    rho_0 = n ##n
    rho_1 = 0.5
    q = 1/ p**1.5
    t = temp
    ndim = len(temp)
    error = np.empty(shape = (Niter,ndim, 4))
    Res_theta = np.empty(shape = (Niter,ndim, p))
    Res_delta = np.empty(shape = (Niter,ndim, p))
    mAcc = np.zeros(ndim)
    swap = np.zeros(ndim-1)
    swap_atem = np.zeros(ndim-1)
    for jj in range(Niter):
        ## Update each coordinate
        for ii in range(ndim):
            
            ## Update delta
            for j in range(p):
                delta_j = delta0[ii,:] + 0
                delta_j[j] = 0
                theta_j = theta0[ii, delta_j == 1]
                if np.sum(theta_j!=0) == 0:
                    r0 = -float('inf')
    #                 print('NA1')
                else:
                    Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
                    B_j =  B[delta_j ==1, :][:, delta_j ==1]
                    r0 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
                delta_j[j] = 1
                theta_j = theta0[ii, delta_j == 1]
                Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
                B_j =  B[delta_j ==1, :][:, delta_j ==1]
                r1 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
                r = (r0 - r1 + np.log(1 - q) -  np.log(q) + np.log(rho_0 / rho_1) / 2 +\
                theta0[ii,j]**2 / 2 * (rho_1 - rho_0)) 
                prob = 1 / (1 + np.exp(r * t[ii]))
                
                delta0[ii,j]  = np.random.uniform(size = 1) <= prob
            ## Update theta

            theta0[ii, delta0[ii,:] == 0] = stats.multivariate_normal.rvs(0, 1/rho_0/t[ii], sum(delta0[ii,:] == 0))
            ind = np.where(delta0[ii,:] != 0)[0]
#             print(ind)
            if len(ind)==0:
                Acc = mAcc[ii]
            else:
                subSighat = A[ind,:][:, ind]
                subB = B[ind,:][:, ind]
                theta_in = theta0[ii, ind]
                theta_out, RQ, Acc = MALA_t(subSighat, subB, sigmasq, np.exp(loggam[ii]) , rho_1, theta_in, t[ii])
                
                theta0[ii, ind] = theta_out.copy()
            if update == True:
                loggam[ii] = loggam[ii] + 1/ (jj**0.6 + 1) * (Acc - 0.3)
            
            mAcc[ii] = mAcc[ii] + 1 / (jj + 1) *(Acc - mAcc[ii])
        if(ndim > 1):
            select = random.sample(list(np.arange(ndim-1)),1)[0]
#             print(loglik(Sighat, sigmasq, rho_0, rho_1, theta0[select,:], delta0[select,], q) )
            alpha = min(1, np.exp( (t[select + 1]- t[select]) * (loglik(A,B, sigmasq, 
                                                                            rho_0, rho_1, theta0[select,:], delta0[select,], q) -
                                                                     loglik(A, B, sigmasq, 
                                                                           rho_0, rho_1, theta0[select+1,:], delta0[select+1,], q)) ))
            
            
            swap_atem[select] = swap_atem[select] + 1
            if np.random.uniform(size=1) <= alpha:
                swap[select] = swap[select] + 1
                aa = theta0[select+1].copy()
                bb = delta0[select+1].copy()
                theta0[select+1] = theta0[select]
                theta0[select] = aa.copy()
                delta0[select+1] = delta0[select]
                delta0[select] = bb.copy()
            
        for ii in range(ndim):
            thetaout =  theta0[ii,:]*(delta0[ii,:]) 
            normcons = np.sqrt(thetaout.dot(B).dot(thetaout))
            thetaout = thetaout /  normcons
	#         error[jj, 0] = norm(np.outer(thetaout, thetaout)  - proj_star1, 'fro')
	#         print(thetaout.T.dot(A).dot(thetaout) / n / 2 / thetaout.dot(B).dot(thetaout))
	#         print(thetaout[:p//2].dot(B[:p//2,:p//2]).dot(thetaout[:p//2]))
            error[jj,ii, 1] = thetaout.T.dot(A).dot(thetaout) / n / 2
            norm1 = norm(thetaout[:p//2])
            norm2 = norm(thetaout[p//2:])
            error[jj,ii, 2] = min((norm(thetaout[:p//2] / norm1 - vx[:,0])**2), (norm(thetaout[:p//2]/ norm1 + vx[:,0])**2))
            error[jj,ii, 3] = min((norm(thetaout[p//2:] / norm2 - vy[:,0])**2), (norm(thetaout[p//2:] / norm2+ vy[:,0])**2))	#         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
            Res_theta[jj,ii, :] = theta0[ii, :] / normcons
            Res_delta[jj,ii, :] = delta0[ii, :]
#         print((Res_theta[jj,:]*Res_delta[jj,:]).dot(B).dot(Res_theta[jj, :]*Res_delta[jj,:]))

    return error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam, theta0, delta0



def Simu_t(A, B, vx, vy, p, n, A0, B0, temp=[1], sigmasq=1, initial = False, Niter = 3000, update = True, loggam = np.zeros(4) -4, rho_0 = 10, batch = False, truncate = False, step = 1, tol = 0, err = [0,0]):
    cri = 0
 #     np.random.seed(seed)
    if initial:
        theta_init = initial[0]
        delta0 = initial[1]
    else:
        theta_init = np.random.normal(size = (p))
        delta0 = np.ones(shape = (p))
        delta0 = stats.bernoulli.rvs(0.5, size = (p))
    ndim = len(temp)
    p1 = len(vx)
    p2 = len(vy)
    vx = vx / norm(vx)
    vy = vy / norm(vy)
    rho_1 = 0.5
    # rho_0 = n 
    q = 1/ p**1.5

    error = np.zeros(shape = (Niter, 5+4))
    Res_theta = np.empty(shape = (Niter, p))
    Res_delta = np.empty(shape = (Niter, p))
    mAcc = 0
    swap = np.zeros(ndim-1)
    swap_atem = np.zeros(ndim-1)
    # theta0 = np.empty(shape = (p))
    # delta0 = np.ones(shape = (p))
    t = temp
    theta0 = theta_init

    aa = 10
    kappa = 0
    c= 0.3


## Here is log phi, i.e. log weights
    phi = np.zeros(shape = ndim)
    phi = - p/2*np.log(temp)
##  initialize log weights
    # jud  = np.NaN
    # while(np.isnan(jud)):
    # 	the, delt = RQ(np.random.normal(size = (p)), A, B, 500,p)
    # 	jud  =  loglik(A, B, sigmasq,  rho_0, rho_1, the, delt, q)
    # phi = -  jud * temp
    # print(phi)

    I =  np.zeros(shape = (Niter))
#     I0 = random.sample(list(np.arange(ndim)),1)[0]
    I0 = 0
    visits = np.zeros(ndim)

    thetaout =  theta0*(delta0) 
    normcons = np.sqrt(thetaout.dot(B).dot(thetaout))
    thetaout = thetaout /  normcons
    error[0, 0] = thetaout.T.dot(A0).dot(thetaout)/thetaout.dot(B0).dot(thetaout) / n / 2
    error[0, 1] = thetaout.T.dot(A).dot(thetaout) / n / 2
    vxhat = thetaout[:p1]
    vyhat = thetaout[p1:]
    norm1 = norm(vxhat)
    norm2 = norm(vyhat)
    error[0, 2] = min((norm(vxhat / norm1 - vx)**2), (norm(vxhat/ norm1 + vx)**2))
    error[0, 3] = min((norm(vyhat / norm2 - vy)**2), (norm(vyhat / norm2+ vy)**2))
    error[0, 5] = np.sum(delta0[:p1][ vx != 0]) /3
    error[0, 6] = np.sum(delta0[p1:][ vy != 0]) /3
    error[0, 7] = 1 - np.sum(delta0[:p1][vx == 0]) /(p1 -3)
    error[0, 8] = 1 - np.sum(delta0[p1:][vy == 0]) /(p2 -3)
    Res_theta[0, :] = theta0 / normcons
    Res_delta[0, :] = delta0
    I[0] = I0
    start = datetime.now()
    for jj in range(Niter-1):
    ## Update delta
        
        if batch:
            sel = random.sample(list(np.arange(p)), batch)
        else:
            sel = np.arange(p)
        for j in sel:
            delta_j = delta0[:] + 0
            delta_j[j] = 0
            theta_j = theta0[ delta_j == 1]
            if np.sum(theta_j!=0) == 0:
                r0 = -float('inf')
    #                 print('NA1')
            else:
                Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
                B_j =  B[delta_j ==1, :][:, delta_j ==1]
                r0 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
            delta_j[j] = 1
            theta_j = theta0[delta_j == 1]
            Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
            B_j =  B[delta_j ==1, :][:, delta_j ==1]
            r1 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
            r = (r0 - r1 + np.log(1 - q) -  np.log(q) + np.log(rho_0 / rho_1) / 2 +\
            theta0[j]**2 / 2 * (rho_1 - rho_0))
            prob = 1 / (1 + np.exp(r * t[I0]))

            delta0[j]  = np.random.uniform(size = 1) <= prob

        ## Update theta

        theta0[ delta0 == 0] = stats.multivariate_normal.rvs(0, 1/rho_0/t[I0], sum(delta0 == 0))
        ind = np.where(delta0 != 0)[0]

        if len(ind)==0:
            Acc = mAcc.copy(),
        else:
            subSighat = A[ind,:][:, ind]
            subB = B[ind,:][:, ind]
            theta_in = theta0[ind]
            theta_out, rq, Acc = MALA_t(subSighat, subB, sigmasq, np.exp(loggam[I0]) , rho_1, theta_in, t[I0], truncate = truncate)

            theta0[ind] = theta_out
        if update == True:
            loggam[I0] = loggam[I0] + step/ (visits[I0] + 1)**0.6 * (Acc - 0.3)

        mAcc = mAcc + 1 / (jj + 1) *(Acc - mAcc)




        thetaout =  theta0*(delta0) 
        normcons = np.sqrt(thetaout.dot(B).dot(thetaout))
        thetaout = thetaout /  normcons
    #         error[jj, 0] = norm(np.outer(thetaout, thetaout)  - proj_star1, 'fro')
        error[jj+1, 0] = thetaout.T.dot(A0).dot(thetaout)/thetaout.dot(B0).dot(thetaout)  / n / 2
        error[jj+1, 1] = thetaout.T.dot(A).dot(thetaout) / n / 2
        vxhat = thetaout[:p1]
        vyhat = thetaout[p1:]
        norm1 = norm(vxhat)
        norm2 = norm(vyhat)
        error[jj+1, 2] = min((norm(vxhat / norm1 - vx)**2), (norm(vxhat/ norm1 + vx)**2))
        error[jj+1, 3] = min((norm(vyhat / norm2 - vy)**2), (norm(vyhat / norm2+ vy)**2))
        
        #         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
        Res_theta[jj+1, :] = theta0 / normcons
        Res_delta[jj+1, :] = delta0
        error[jj+1, 5] = np.sum(delta0[:p1][ vx != 0]) /3
        error[jj+1, 6] = np.sum(delta0[p1:][vy != 0]) /3
        error[jj+1, 7] = 1 - np.sum(delta0[:p1][ vx == 0]) /(p1 -3)
        error[jj+1, 8] = 1 - np.sum(delta0[p1:][ vy == 0]) /(p2 -3)
        ##Update temperature index
        if I0 == 0:
            I_prop = 1
            Acc = (loglik(A, B, sigmasq,  rho_0, rho_1, theta0, delta0, q) *(t[I_prop]-  t[I0]))- phi[I_prop] + phi[I0]
            Acc = min(1, np.exp(Acc)*0.5)
        elif I0 == ndim - 1:
            I_prop = ndim - 2
            Acc = (loglik(A, B, sigmasq,  rho_0, rho_1, theta0, delta0, q) *(t[I_prop]-  t[I0]))- phi[I_prop] + phi[I0]
            Acc = min(1, np.exp(Acc)*0.5)
        else:
            if np.random.uniform(size=1)<=0.5:
                I_prop = I0 - 1
            else:
                I_prop = I0 + 1
            Acc = (loglik(A, B, sigmasq,  rho_0, rho_1, theta0, delta0, q) *(t[I_prop]-  t[I0]))- phi[I_prop] + phi[I0]
            if I_prop == 0:
                Acc = min(1, np.exp(Acc)*2)
            elif I_prop == ndim-1:
                Acc = min(1, np.exp(Acc)*2)
            else:
                Acc = min(1, np.exp(Acc))
        if np.random.uniform(size=1) <= Acc:
            I0 = I_prop
        I[jj+1] = I0
    #     phi[I0] = phi[I0] * (1 + 2**(-aa))
    #     judge = np.zeros(ndim)
    #     for ii in range(ndim):
    #         judge[ii] = 1/max(0,jj+1- kappa) * np.sum(I[kappa+1:jj+2]==ii) - 1/ndim
    #     if np.max(np.abs(judge)) <= 0.3/ndim:
    #         kappa = jj +1
    #         aa = aa + 1
        phi[I0] = phi[I0]+ aa
        visits[I0] = visits[I0]+1
        if np.max(np.abs(visits/(jj+1) - 1/ ndim)) < 0.4 / ndim:
            aa = aa/2
#             print(jj, visits, "time when aa changes")
            visits = np.zeros(ndim)
        error[jj+1, 4] = (datetime.now() - start).total_seconds()
        # print(jj, I0, loglik(A, B, sigmasq,  rho_0, rho_1, theta0, delta0, q))
       
        indices = 0
        if jj >= indices:
            if (np.mean(error[(jj-indices+1):(jj+2), 2]) - err[0]) < tol and (np.mean(error[(jj-indices+1): (jj+2), 3]) - err[1]) < tol:
                break
    return error[:(jj+1), :], Res_theta[:(jj+1), :], Res_delta[:(jj+1), :], swap_atem, swap, mAcc, loggam, I[:(jj+1)], phi,aa, visits

def Simu_var(A, B, p1, p2, n, temp=[1], sigmasq=1, initial = False, Niter = 3000, update = True, loggam = -5, rho_0 = 10, batch = False):
 #     np.random.seed(seed)
    p = p1+p2
    if initial:
        theta_init = initial[0]
        delta0 = initial[1]
    else:
        theta_init = np.random.normal(size = (p))
        delta0 = np.ones(shape = (p))
        delta0 = stats.bernoulli.rvs(0.5, size = (p))
    ndim = len(temp)

    rho_1 = 0.5
    # rho_0 = n 
    q = 1/ p**1.5
    


    #     Niter = 4000

    # loggam = -5
    error = np.empty(shape = (Niter, 4))
    Res_theta = np.empty(shape = (Niter, p))
    Res_delta = np.empty(shape = (Niter, p))
    mAcc = 0
    swap = np.zeros(ndim-1)
    swap_atem = np.zeros(ndim-1)
    theta0 = np.empty(shape = (p))
    # delta0 = np.ones(shape = (p))
    t = temp
    theta0 = theta_init / np.sqrt((theta_init.T.dot(B).dot(theta_init)))

    aa = 10
    kappa = 0
    c= 0.3


## Here is log phi, i.e. log weights
    phi = np.zeros(shape = ndim)
    phi = - p/2*np.log(temp)
##  initialize log weights
    # jud  = np.NaN
    # while(np.isnan(jud)):
    # 	the, delt = RQ(np.random.normal(size = (p)), A, B, 500,p)
    # 	jud  =  loglik(A, B, sigmasq,  rho_0, rho_1, the, delt, q)
    # phi = -  jud * temp


    # print(phi)

    I =  np.zeros(shape = (Niter))
#     I0 = random.sample(list(np.arange(ndim)),1)[0]
    I0 = 0
    visits = np.zeros(ndim)

    thetaout =  theta0*(delta0) 
    normcons = np.sqrt(thetaout.dot(B).dot(thetaout))
    thetaout = thetaout /  normcons
    error[0, 1] = thetaout.T.dot(A).dot(thetaout) / n / 2
    Res_theta[0, :] = theta0 / normcons
    Res_delta[0, :] = delta0
    I[0] = I0

    for jj in range(Niter-1):
    ## Update delta
        if batch:
            sel = random.sample(list(np.arange(p)), batch)
        else:
            sel = np.arange(p)
        for j in sel:
            delta_j = delta0[:] + 0
            delta_j[j] = 0
            theta_j = theta0[ delta_j == 1]
            if np.sum(theta_j!=0) == 0:
                r0 = -float('inf')
    #                 print('NA1')
            else:
                Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
                B_j =  B[delta_j ==1, :][:, delta_j ==1]
                r0 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
            delta_j[j] = 1
            theta_j = theta0[delta_j == 1]
            Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
            B_j =  B[delta_j ==1, :][:, delta_j ==1]
            r1 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
            r = (r0 - r1 + np.log(1 - q) -  np.log(q) + np.log(rho_0 / rho_1) / 2 +\
            theta0[j]**2 / 2 * (rho_1 - rho_0))
            prob = 1 / (1 + np.exp(r * t[I0]))

            delta0[j]  = np.random.uniform(size = 1) <= prob

        ## Update theta

        theta0[ delta0 == 0] = stats.multivariate_normal.rvs(0, 1/rho_0/t[I0], sum(delta0 == 0))
        ind = np.where(delta0 != 0)[0]

        if len(ind)==0:
            Acc = mAcc
        else:
            subSighat = A[ind,:][:, ind]
            subB = B[ind,:][:, ind]
            theta_in = theta0[ind]
            theta_out, rq, Acc = MALA_t(subSighat, subB, sigmasq, np.exp(loggam[I0]) , rho_1, theta_in, t[I0])

            theta0[ind] = theta_out
        if update == True:
            loggam[I0] = loggam[I0] + 1/ (jj**0.6 + 1) * (Acc - 0.3)

        mAcc = mAcc + 1 / (jj + 1) *(Acc - mAcc)




        thetaout =  theta0*(delta0) 
        normcons = np.sqrt(thetaout.dot(B).dot(thetaout))
        thetaout = thetaout /  normcons
    #         error[jj, 0] = norm(np.outer(thetaout, thetaout)  - proj_star1, 'fro')
    #         print(thetaout.T.dot(A).dot(thetaout) / n / 2 / thetaout.dot(B).dot(thetaout))
    #         print(thetaout[:p//2].dot(B[:p//2,:p//2]).dot(thetaout[:p//2]))
        error[jj+1, 1] = thetaout.T.dot(A).dot(thetaout) / n / 2
    #         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
        Res_theta[jj+1, :] = theta0 / normcons
        Res_delta[jj+1, :] = delta0

        ##Update temperature index
        if I0 == 0:
            I_prop = 1
            Acc = (loglik(A, B, sigmasq,  rho_0, rho_1, theta0, delta0, q) *(t[I_prop]-  t[I0]))- phi[I_prop] + phi[I0]
            Acc = min(1, np.exp(Acc)*0.5)
        elif I0 == ndim - 1:
            I_prop = ndim - 2
            Acc = (loglik(A, B, sigmasq,  rho_0, rho_1, theta0, delta0, q) *(t[I_prop]-  t[I0]))- phi[I_prop] + phi[I0]
            Acc = min(1, np.exp(Acc)*0.5)
        else:
            if np.random.uniform(size=1)<=0.5:
                I_prop = I0 - 1
            else:
                I_prop = I0 + 1
            Acc = (loglik(A, B, sigmasq,  rho_0, rho_1, theta0, delta0, q) *(t[I_prop]-  t[I0]))- phi[I_prop] + phi[I0]
            if I_prop == 0:
                Acc = min(1, np.exp(Acc)*2)
            elif I_prop == ndim-1:
                Acc = min(1, np.exp(Acc)*2)
            else:
                Acc = min(1, np.exp(Acc))
        if np.random.uniform(size=1) <= Acc:
            I0 = I_prop
        I[jj+1] = I0
    #     phi[I0] = phi[I0] * (1 + 2**(-aa))
    #     judge = np.zeros(ndim)
    #     for ii in range(ndim):
    #         judge[ii] = 1/max(0,jj+1- kappa) * np.sum(I[kappa+1:jj+2]==ii) - 1/ndim
    #     if np.max(np.abs(judge)) <= 0.3/ndim:
    #         kappa = jj +1
    #         aa = aa + 1
        phi[I0] = phi[I0]+ aa
        visits[I0] = visits[I0]+1
        if np.max(np.abs(visits/(jj+1) - 1/ ndim)) < 0.4 / ndim:
            aa = aa/2
            # print(jj, visits, "time when aa changes")
            visits = np.zeros(ndim)
        # print(jj)



        print(jj, I0, loglik(A, B, sigmasq,  rho_0, rho_1, theta0, delta0, q))

        
    return error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam, I, phi,aa, visits


def ParaBayes2(A, B, vx, vy, p, n, temp=[1], sigmasq=1, initial = True, Niter = 3000, update = True, loggam = [-13]):
#     np.random.seed(100)
    vx = vx / norm(vx)
    vy = vy / norm(vy)
    theta_init = np.random.normal(size = (p, 1))
    if initial == True:
        Niter1 =2000
        thetahat, delta = RQ(theta_init, A, B, Niter1,p)
    else:
        thetahat = theta_init / np.sqrt((theta_init.T.dot(B).dot(theta_init)))
        delta = np.ones(p)
    ndim = len(temp)
#     error = np.empty(shape = (Niter, 4))
#     Res_theta = np.empty(shape = (Niter, p))
#     Res_delta = np.empty(shape = (Niter, p))
#     mAcc = np.zeros(ndim)
#     swap = np.zeros(ndim-1)
#     swap_atem = np.zeros(ndim-1)
    theta0 = np.empty(shape = (ndim, p))
    delta0 = np.ones(shape = (ndim, p))
    theta1 = np.random.normal(size = (ndim, p))
    delta1 = np.ones(shape = (ndim, p))
    theta2 = np.random.normal(size = (ndim, p))
    delta2 = np.ones(shape = (ndim, p))
    
    
    for i in range(ndim):
        theta0[i] = thetahat[:,0]
#         delta0[i] = delta
        theta1[i] = theta1[i] /  np.sqrt((theta1[i].T.dot(B).dot(theta1[i])))
        theta2[i] = theta2[i] /  np.sqrt((theta2[i].T.dot(B).dot(theta2[i])))
#         t[i] = 1 / temp**i
#         t[i] = 1 / (1+i*temp)
    
    theta_c = (theta0+theta1+theta2)/3
    error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam = mcmc(theta0, delta0, A, B, vx, vy, p, n, temp=[1, 0.9, 0.8], sigmasq=1, initial = True, Niter = 2000, update = True, loggam = [-5])
#     error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam = mcmc(Res_theta[-1:,:], Res_delta[-1:,:], A, B, vx, vy, p, n, temp=[1], sigmasq=1.6, initial = True, Niter = 500, update = True, loggam = [-5])
#     error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam = mcmc(Res_theta[-1:,:], Res_delta[-1:,:], A, B, vx, vy, p, n, temp=[1], sigmasq=1.5, initial = True, Niter = 500, update = True, loggam = [-5])
#     error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam = mcmc(Res_theta[-1:,:], Res_delta[-1:,:], A, B, vx, vy, p, n, temp=[1], sigmasq=1.4, initial = True, Niter = 500, update = True, loggam = [-5])
#     error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam = mcmc(Res_theta[-1:,:], Res_delta[-1:,:], A, B, vx, vy, p, n, temp=[1], sigmasq=1, initial = True, Niter = 500, update = True, loggam = [-5])
#     theta0[0], delta0[0] = RQ(Res_theta[-1], A, B, 1000)
#     error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam = mcmc(Res_theta[-1:,:], Res_delta[-1:,:], A, B, vx, vy, p, n, temp=[1], sigmasq=1.7, initial = True, Niter = 2000, update = True, loggam = [-5])

    return error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam


## use inverse temperature here



def mix_time_PT( A, B, vx, vy, p, n,lag  =  1, temp=[1], sigmasq=1, initial = True, Niter = 3000, update = False, loggam = [-3.94685961, -4.01251142, -4.93310664, -3.26939622]):
#     np.random.seed(100)
    ndim = len(temp)
    vx = vx / norm(vx)
    vy = vy / norm(vy)



    #     sigmasq = 0.5 / vartheta
    #     sigmasq = 1 / np.linalg.eig(Sighat)[0][1]
    rho_1 = 0.5
    rho_0 = n ##n
    q = 1/ p**1.5


    #     Niter = 4000

#     loggam =[-3.94685961, -4.01251142, -4.93310664, -3.26939622]
    error = np.empty(shape = (2, Niter,ndim, 4))
    Res_theta = np.zeros(shape = (2, Niter, ndim, p))
    Res_delta = np.zeros(shape = (2, Niter, ndim, p))
    mAcc = np.zeros((2, ndim))
    swap = np.zeros((2, ndim-1))
    swap_atem = np.zeros((2, ndim-1))
    theta0 = np.zeros(shape = (2, ndim, p))
    delta0 = np.ones(shape = (2, ndim, p))
    t = temp

    difference = np.zeros(shape = (Niter, ndim))

    ## Initialize X_0 and  Y_0
    theta_init = np.random.normal(size = (2, ndim,p))

    for i in range(ndim):
        for k in range(2):
            theta0[k, i] = theta_init[k, i] / np.sqrt((theta_init[k, i].T.dot(B).dot(theta_init[k, i])))
    #         t[i] = 1 / temp**i
    #         t[i] = 1 / (1+i*temp)


    ## Initialize X1, ...,  X_lag

    error[0,:lag,:,:], Res_theta[0,:lag,:,:], Res_delta[0,:lag,:,:], swap_atem[0,:], swap[0,:], mAcc[0,:], loggam, theta0[0,:,:], delta0[0,:,:] = mcmc(theta0[0,:], delta0[0,:,:], A, B, vx, vy, p, n, temp=t, sigmasq=sigmasq, initial = False, Niter = lag, update = update, loggam = loggam)


    ## Coupling 
    for jj in range(lag, Niter):
        for ii in range(ndim):
            for j in range(p):
                ## For delta
                U = np.random.uniform(size = 1)

                delta_j = delta0[0,ii,:] + 0
                delta_j[j] = 0
                theta_j = theta0[0, ii, delta_j == 1]
                if np.sum(theta_j!=0) == 0:
                    r0 = -float('inf')
    #                 print(ii,'NA1')
                else:
                    Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
                    B_j =  B[delta_j ==1, :][:, delta_j ==1]
                    r0 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
                delta_j[j] = 1
                theta_j = theta0[0,ii, delta_j == 1]
                Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
                B_j =  B[delta_j ==1, :][:, delta_j ==1]
                r1 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
                r = (r0 - r1 + np.log(1 - q) -  np.log(q) + np.log(rho_0 / rho_1) / 2 +\
                theta0[0,ii,j]**2 / 2 * (rho_1 - rho_0)) 
                prob = 1 / (1 + np.exp(r * t[ii]))

                delta0[0,ii,j]  = U <= prob
    #             if j == 0:
    #                 print(0,jj,ii,prob,delta0[0,ii,j])

                delta_j = delta0[1,ii,:] + 0
                delta_j[j] = 0
                theta_j = theta0[1, ii, delta_j == 1]
                if np.sum(theta_j!=0) == 0:
                    r0 = -float('inf')
    #                 print(ii,'NA1')
                else:
                    Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
                    B_j =  B[delta_j ==1, :][:, delta_j ==1]
                    r0 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
                delta_j[j] = 1
                theta_j = theta0[1,ii, delta_j == 1]
                Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
                B_j =  B[delta_j ==1, :][:, delta_j ==1]
                r1 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
                r = (r0 - r1 + np.log(1 - q) -  np.log(q) + np.log(rho_0 / rho_1) / 2 +\
                theta0[1,ii,j]**2 / 2 * (rho_1 - rho_0)) 
                prob = 1 / (1 + np.exp(r * t[ii]))
                delta0[1, ii,j]  = U <= prob
    #             if j == 0:
    #                 print(1,jj,ii,prob,delta0[1,ii,j])
            ## split apart indexes into 4 groups
            index00 = np.nonzero((delta0[0,ii,:] == 0) & (delta0[1,ii,:] == 0))[0]
            index01 = np.nonzero((delta0[0,ii,:] == 0) & (delta0[1,ii,:] == 1))[0]
            index10 = np.nonzero((delta0[0,ii,:] == 1) & (delta0[1,ii,:] == 0))[0]
            index11 = np.nonzero((delta0[0,ii,:] == 1) & (delta0[1,ii,:] == 1))[0]

            ## update 00
            n1 = len(index00)
            theta0[0, ii, index00] = stats.multivariate_normal.rvs(0, 1/rho_0/t[ii], n1)
            theta0[1, ii, index00] = theta0[0, ii, index00].copy()

            ## update 01
            n2 = len(index01)
            if n2 == 0:
                Acc = mAcc[:, ii]
            else:
                Z = np.random.normal(size = (n2))
                theta0[0, ii, index01] = Z * np.sqrt( 1/rho_0/t[ii])

                tau = np.exp(loggam[ii])
                subSighat = A[index01,:][:, index01]
                subB = B[index01,:][:, index01]
                theta_in = theta0[1, ii, index01]
                gradient = gradient_computer(theta_in, subSighat, subB, sigmasq, rho_1) * t[ii]
                thetahat_prop = theta_in + tau* gradient + np.sqrt(2* tau ) * Z
                RQ = 1 / sigmasq * theta_in.T.dot(subSighat).dot(theta_in)/ theta_in.dot(subB).dot(theta_in) *t[ii]
                RQ_prop = 1/ sigmasq * thetahat_prop.T.dot(subSighat).dot(thetahat_prop)/ thetahat_prop.dot(subB).dot(thetahat_prop) *t[ii]
                dif = thetahat_prop - theta_in
                Acc = RQ_prop - RQ - (rho_1 /2) * (norm(thetahat_prop)**2 - norm(theta_in)**2) * t[ii] -  (np.sum(
                    (-dif - tau * gradient_computer(thetahat_prop, subSighat, subB, sigmasq, rho_1) * t[ii])**2)- np.sum((dif - tau * gradient)**2))/4/tau
                Acc = min(1, np.exp(Acc))
            #     print(Acc)
                if np.random.uniform(size=1) <= Acc:
                    theta_in = thetahat_prop.copy()
                    RQ = RQ_prop.copy()
                theta0[1, ii, index01] = theta_in.copy()
            mAcc[:,ii] = mAcc[:,ii] + 1 / (jj + 1) *(Acc - mAcc[:,ii])  


            ## update 10
            n3 = len(index10)
            if n3 == 0:
                Acc = mAcc[:, ii]
            else:
                Z = np.random.normal(size = (n3))
                theta0[1, ii, index10] = Z * np.sqrt( 1/rho_0/t[ii])
                tau = np.exp(loggam[ii])
                subSighat = A[index10,:][:, index10]
                subB = B[index10,:][:, index10]
                theta_in = theta0[0, ii, index10]
                gradient = gradient_computer(theta_in, subSighat, subB, sigmasq, rho_1) * t[ii]
                thetahat_prop = theta_in + tau* gradient + np.sqrt(2* tau ) * Z
                RQ = 1 / sigmasq * theta_in.T.dot(subSighat).dot(theta_in)/ theta_in.dot(subB).dot(theta_in) *t[ii]
                RQ_prop = 1/ sigmasq * thetahat_prop.T.dot(subSighat).dot(thetahat_prop)/ thetahat_prop.dot(subB).dot(thetahat_prop) *t[ii]
                dif = thetahat_prop - theta_in
                Acc = RQ_prop - RQ - (rho_1 /2) * (norm(thetahat_prop)**2 - norm(theta_in)**2) * t[ii] -  (np.sum(
                    (-dif - tau * gradient_computer(thetahat_prop, subSighat, subB, sigmasq, rho_1) * t[ii])**2)- np.sum((dif - tau * gradient)**2))/4/tau
                Acc = min(1, np.exp(Acc))
            #     print(Acc)
                if np.random.uniform(size=1) <= Acc:
                    thetain = thetahat_prop.copy()
                    RQ = RQ_prop.copy()
                theta0[0, ii, index10] = theta_in   .copy()
            mAcc[:,ii] = mAcc[:,ii] + 1 / (jj + 1) *(Acc - mAcc[:,ii])  

            ## Update 11
            ## Sample proposals by reflection-maximal coupling
            n4 = len(index11)
            tau = np.exp(loggam[ii])
            subSighat = A[index11,:][:, index11]
            subB = B[index11,:][:, index11]
            z = (theta0[0, ii, index11] - theta0[1, ii, index11] + tau * (gradient_computer(theta0[0, ii, index11], subSighat, subB, sigmasq, rho_1) - gradient_computer(theta0[1, ii, index11], subSighat, subB, sigmasq, rho_1) )* t[ii]) / np.sqrt(2*tau)
            X_d = np.random.normal(size = (n4))
            if norm(z)==0:
                Y_d = X_d.copy()
            else:
                e = z / norm(z)
                W = np.random.uniform(size=1)
                if stats.multivariate_normal.pdf(X_d, np.zeros(n4)) * W <= stats.multivariate_normal.pdf(X_d+z, np.zeros(n4)) :
                    Y_d = X_d + z
                else:
                    Y_d = X_d - 2*(e.T.dot(X_d)) * e
            X_prop = X_d * np.sqrt(2*tau) + theta0[0, ii, index11] + tau * gradient_computer(theta0[0, ii, index11], subSighat, subB, sigmasq, rho_1)
            Y_prop= Y_d * np.sqrt(2*tau) + theta0[1, ii, index11] + tau * gradient_computer(theta0[1, ii, index11], subSighat, subB, sigmasq, rho_1)
            U = np.random.uniform(size=1)

            theta_in = theta0[0, ii, index11].copy()
            gradient = gradient_computer(theta_in, subSighat, subB, sigmasq, rho_1) * t[ii]
            thetahat_prop = X_prop.copy()
            if theta_in.dot(subB).dot(theta_in) == 0:
                RQ = -float('inf')
            else:
                RQ = 1 / sigmasq * theta_in.T.dot(subSighat).dot(theta_in)/ theta_in.dot(subB).dot(theta_in) *t[ii]
            if thetahat_prop.dot(subB).dot(thetahat_prop) != 0:
                RQ_prop = 1/ sigmasq * thetahat_prop.T.dot(subSighat).dot(thetahat_prop)/ thetahat_prop.dot(subB).dot(thetahat_prop) *t[ii]
                dif = thetahat_prop - theta_in
                Acc = RQ_prop - RQ - (rho_1 /2) * (norm(thetahat_prop)**2 - norm(theta_in)**2) * t[ii] -  (np.sum(
                    (-dif - tau * gradient_computer(thetahat_prop, subSighat, subB, sigmasq, rho_1) * t[ii])**2)- np.sum((dif - tau * gradient)**2))/4/tau
                Acc = min(1, np.exp(Acc))
                if U <= Acc:
                    theta0[0, ii, index11] = thetahat_prop.copy()

            theta_in = theta0[1, ii, index11]
            gradient = gradient_computer(theta_in, subSighat, subB, sigmasq, rho_1) * t[ii]
            thetahat_prop = Y_prop.copy()
            if theta_in.dot(subB).dot(theta_in) == 0:
                RQ = -float('inf')
            else:
                RQ = 1 / sigmasq * theta_in.T.dot(subSighat).dot(theta_in)/ theta_in.dot(subB).dot(theta_in) *t[ii]
            if thetahat_prop.dot(subB).dot(thetahat_prop) != 0:
                RQ_prop = 1/ sigmasq * thetahat_prop.T.dot(subSighat).dot(thetahat_prop)/ thetahat_prop.dot(subB).dot(thetahat_prop) *t[ii]
                dif = thetahat_prop - theta_in
                Acc = RQ_prop - RQ - (rho_1 /2) * (norm(thetahat_prop)**2 - norm(theta_in)**2) * t[ii] -  (np.sum(
                    (-dif - tau * gradient_computer(thetahat_prop, subSighat, subB, sigmasq, rho_1) * t[ii])**2)- np.sum((dif - tau * gradient)**2))/4/tau
                Acc = min(1, np.exp(Acc))
                if U <= Acc:
                    theta0[1, ii, index11] = thetahat_prop.copy()

        ##  Swap temp
        if(ndim > 1):
            select = random.sample(list(np.arange(ndim-1)),1)[0]
            U = np.random.uniform(size=1)


            alpha = min(1, np.exp( (t[select + 1]- t[select]) * (loglik(A,B, sigmasq, 
                                                                                rho_0, rho_1, theta0[0,select,:], delta0[0,select,], q) -
                                                                         loglik(A, B, sigmasq, 
                                                                               rho_0, rho_1, theta0[0,select+1,:], delta0[0,select+1,], q)) ))

            swap_atem[0,select] = swap_atem[0,select] + 1
            if U <= alpha:
                swap[0,select] = swap[0,select] + 1
                aa = theta0[0,select+1,:].copy()
                bb = delta0[0,select+1,:].copy()
                theta0[0,select+1,:] = theta0[0,select,:]
                theta0[0,select,:] = aa.copy()
                delta0[0,select+1,:] = delta0[0,select,:]
                delta0[0,select,:] = bb.copy()

            alpha = min(1, np.exp( (t[select + 1]- t[select]) * (loglik(A,B, sigmasq, 
                                                                                rho_0, rho_1, theta0[1,select,:], delta0[1,select,], q) -
                                                                         loglik(A, B, sigmasq, 
                                                                               rho_0, rho_1, theta0[1,select+1,:], delta0[1,select+1,], q)) ))

            swap_atem[1,select] = swap_atem[1,select] + 1
            if U <= alpha:
                swap[1,select] = swap[1,select] + 1
                aa = theta0[1,select+1,:].copy()
                bb = delta0[1,select+1,:].copy()
                theta0[1,select+1,:] = theta0[1,select,:]
                theta0[1,select,:] = aa.copy()
                delta0[1,select+1,:] = delta0[1,select,:]
                delta0[1,select,:] = bb.copy()

        ## Record values
        for ii in range(ndim):
            thetaout =  theta0[0,ii,:]*(delta0[0,ii,:]) 
            normcons = np.sqrt(thetaout.dot(B).dot(thetaout))
            thetaout = thetaout /  normcons
    #         error[jj, 0] = norm(np.outer(thetaout, thetaout)  - proj_star1, 'fro')
    #         print(thetaout.T.dot(A).dot(thetaout) / n / 2 / thetaout.dot(B).dot(thetaout))
    #         print(thetaout[:p//2].dot(B[:p//2,:p//2]).dot(thetaout[:p//2]))
            error[0,jj,ii, 1] = thetaout.T.dot(A).dot(thetaout) / n / 2
            norm1 = norm(thetaout[:p//2])
            norm2 = norm(thetaout[p//2:])
            error[0,jj,ii, 2] = min((norm(thetaout[:p//2] / norm1 - vx[:,0])**2), (norm(thetaout[:p//2]/ norm1 + vx[:,0])**2))
            error[0,jj,ii, 3] = min((norm(thetaout[p//2:] / norm2 - vy[:,0])**2), (norm(thetaout[p//2:] / norm2+ vy[:,0])**2))    #         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
            Res_theta[0,jj,ii, :] = theta0[0,ii, :] / normcons
            Res_delta[0,jj, ii, :] = delta0[0,ii, :]

            thetaout =  theta0[1,ii,:]*(delta0[1,ii,:]) 
            normcons = np.sqrt(thetaout.dot(B).dot(thetaout))
            thetaout = thetaout /  normcons
    #         error[jj, 0] = norm(np.outer(thetaout, thetaout)  - proj_star1, 'fro')
    #         print(thetaout.T.dot(A).dot(thetaout) / n / 2 / thetaout.dot(B).dot(thetaout))
    #         print(thetaout[:p//2].dot(B[:p//2,:p//2]).dot(thetaout[:p//2]))
            error[1,jj-lag,ii, 1] = thetaout.T.dot(A).dot(thetaout) / n / 2
            norm1 = norm(thetaout[:p//2])
            norm2 = norm(thetaout[p//2:])
            error[0,jj-lag,ii, 2] = min((norm(thetaout[:p//2] / norm1 - vx[:,0])**2), (norm(thetaout[:p//2]/ norm1 + vx[:,0])**2))
            error[0,jj-lag,ii, 3] = min((norm(thetaout[p//2:] / norm2 - vy[:,0])**2), (norm(thetaout[p//2:] / norm2+ vy[:,0])**2))    #         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
    #         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
            Res_theta[1,jj-lag,ii, :] = theta0[1,ii, :] / normcons
            Res_delta[1,jj-lag,ii, :] = delta0[1,ii, :]

            difference[jj,ii] = norm(Res_theta[1,jj-lag,ii, :] - Res_theta[0,jj,ii, :] ) + np.sum(np.abs(Res_delta[1,jj-lag,ii, :]- Res_delta[0,jj,ii, :]))

        if np.sum(difference[jj]) < 0.01:
            Niter = jj
            return error[:,:jj,:,:], Res_theta[:,:jj,:,:], Res_delta[:,:jj,:,:], swap_atem[:,:jj], swap[:,:jj], mAcc[:,:jj], loggam, Niter
    return error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam, Niter

def ST_mix(A, B, vx, vy, p, n, lag = 1, temp=[1], sigmasq=1, initial = False, Niter = 3000, update = True, loggam = -5, rho_0 = 10, batch = False, truncate = False):




    vx = vx / norm(vx)
    vy = vy / norm(vy)
    ndim = len(temp)


    rho_1 = 0.5
    q = 1/ p**1.5


    error = np.empty(shape = (2, Niter, 4))
    Res_theta = np.zeros(shape = (2, Niter, p))
    Res_delta = np.zeros(shape = (2, Niter, p))
    mAcc = np.zeros((2))
    swap = np.zeros((2, ndim-1))
    swap_atem = np.zeros((2, ndim-1))
    theta0 = np.zeros(shape = (2, p))
    delta0 = np.ones(shape = (2, p))
    delta0 = stats.bernoulli.rvs(0.5, size = (2,p))
#    delta0[0, [0, 5, 10]] = 0
#    delta0[1, [0, 5, 10]] = 0
#    delta0 = np.zeros(shape = (2, p))
#    delta0[0,100] = 1
#    delta0[1,100] = 1
#    print(delta0)
    t = temp

    difference = np.zeros(shape = (Niter))

    ## Initialize X_0 and  Y_0
    theta_init = np.random.normal(size = (2, p))


    for k in range(2):
        theta0[k] = theta_init[k] / np.sqrt((theta_init[k].T.dot(B).dot(theta_init[k])))
    #         t[i] = 1 / temp**i
    #         t[i] = 1 / (1+i*temp)
    aa = [10,10]
    kappa = 0
    c = 0.3

    ## Here is log phi, i.e. log weights
    phi = np.zeros(shape = (2,ndim))
    phi[0] = - p/2*np.log(temp)
    phi[1] = - p/2*np.log(temp)

    I =  np.zeros(shape = (2, Niter))
    I0 = [0, 0]

    visits = np.zeros((2, ndim))

    thetaout =  theta0[0]*(delta0[0]) 
    normcons = np.sqrt(thetaout.dot(B).dot(thetaout))
    thetaout = thetaout /  normcons
    error[0, 0, 1] = thetaout.T.dot(A).dot(thetaout) / n / 2
    norm1 = norm(thetaout[:p//2])
    norm2 = norm(thetaout[p//2:])
    error[0,0, 2] = min((norm(thetaout[:p//2] / norm1 - vx[:,0])**2), (norm(thetaout[:p//2]/ norm1 + vx[:,0])**2))
    error[0,0, 3] = min((norm(thetaout[p//2:] / norm2 - vy[:,0])**2), (norm(thetaout[p//2:] / norm2+ vy[:,0])**2))	#         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
    Res_theta[0, 0, :] = theta0[0] / normcons
    Res_delta[0, 0, :] = delta0[0]
    I[0, 0] = I0[0]

    thetaout =  theta0[1]*(delta0[1]) 
    normcons = np.sqrt(thetaout.dot(B).dot(thetaout))
    thetaout = thetaout /  normcons
    error[1, 0, 1] = thetaout.T.dot(A).dot(thetaout) / n / 2
    norm1 = norm(thetaout[:p//2])
    norm2 = norm(thetaout[p//2:])
    error[1,0, 2] = min((norm(thetaout[:p//2] / norm1 - vx[:,0])**2), (norm(thetaout[:p//2]/ norm1 + vx[:,0])**2))
    error[1,0, 3] = min((norm(thetaout[p//2:] / norm2 - vy[:,0])**2), (norm(thetaout[p//2:] / norm2+ vy[:,0])**2))	#         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
    Res_theta[1, 0, :] = theta0[1] / normcons
    Res_delta[1, 0, :] = delta0[1]
    I[1, 0] = I0[1]
    ## Initialize X1, ...,  X_lag

    error[0,1:(lag+1),:], Res_theta[0,1:(lag+1),:], Res_delta[0,1:(lag+1),:], swap_atem[0,:], swap[0,:], mAcc[0], loggam_cc, \
    I[0,1:(lag+1)], phi[0,:], aa[0],visits[0]= \
    Simu_t(A, B, vx, vy, p, n, temp=t, sigmasq=sigmasq,initial = (theta0[0,:], delta0[0,:]),Niter = lag, update = update, rho_0 = rho_0, loggam = loggam, batch = batch, truncate = truncate)

    ## Coupling 
    for jj in range(lag+1, Niter-1):
        if batch:
            sel = random.sample(list(np.arange(p)), batch)
        else:
            sel = np.arange(p)
        for j in sel:
            ## For delta
            U = np.random.uniform(size = 1)

            delta_j = delta0[0,:] + 0
            delta_j[j] = 0
            theta_j = theta0[0, delta_j == 1]
            if np.sum(theta_j!=0) == 0:
                r0 = -float('inf')
    #                 print(ii,'NA1')
            else:
                Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
                B_j =  B[delta_j ==1, :][:, delta_j ==1]
                r0 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
            delta_j[j] = 1
            theta_j = theta0[0, delta_j == 1]
            Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
            B_j =  B[delta_j ==1, :][:, delta_j ==1]
            r1 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
            r = (r0 - r1 + np.log(1 - q) -  np.log(q) + np.log(rho_0 / rho_1) / 2 +\
            theta0[0,j]**2 / 2 * (rho_1 - rho_0)) 
            prob = 1 / (1 + np.exp(r * t[I0[0]]))

            delta0[0,j]  = U <= prob

            delta_j = delta0[1,:] + 0
            delta_j[j] = 0
            theta_j = theta0[1, delta_j == 1]
            if np.sum(theta_j!=0) == 0:
                r0 = -float('inf')
    #                 print(ii,'NA1')
            else:
                Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
                B_j =  B[delta_j ==1, :][:, delta_j ==1]
                r0 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
            delta_j[j] = 1
            theta_j = theta0[1, delta_j == 1]
            Sigma_j = A[delta_j ==1, :][:, delta_j ==1]
            B_j =  B[delta_j ==1, :][:, delta_j ==1]
            r1 = theta_j.T.dot(Sigma_j).dot(theta_j)/sigmasq/ theta_j.dot(B_j).dot(theta_j)
            r = (r0 - r1 + np.log(1 - q) -  np.log(q) + np.log(rho_0 / rho_1) / 2 +\
            theta0[1,j]**2 / 2 * (rho_1 - rho_0)) 
            prob = 1 / (1 + np.exp(r * t[I0[1]]))
            delta0[1, j]  = U <= prob


        ## split apart indexes into 4 groups
        index00 = np.nonzero((delta0[0,:] == 0) & (delta0[1,:] == 0))[0]
        index01 = np.nonzero((delta0[0,:] == 0) & (delta0[1,:] == 1))[0]
        index10 = np.nonzero((delta0[0,:] == 1) & (delta0[1,:] == 0))[0]
        index11 = np.nonzero((delta0[0,:] == 1) & (delta0[1,:] == 1))[0]

        ## update 00
        n1 = len(index00)
        z = stats.multivariate_normal.rvs(0, 1/rho_0, n1)
        theta0[0, index00] = z / np.sqrt(t[I0[0]])
        theta0[1, index00] = z / np.sqrt(t[I0[1]])

        ## update 01
        n2 = len(index01)
        if n2 == 0:
            Acc = mAcc[:]
        else:
            Z = np.random.normal(size = (n2))
            theta0[0, index01] = Z * np.sqrt( 1/rho_0/t[I0[0]])

            tau = np.exp(loggam[I0[1]])
            subSighat = A[index01,:][:, index01]
            subB = B[index01,:][:, index01]
            theta_in = theta0[1, index01]
            gradient = gradient_computer(theta_in, subSighat, subB, sigmasq, rho_1) * t[I0[1]]
            thetahat_prop = theta_in + tau* gradient + np.sqrt(2* tau ) * Z
            RQ = 1 / sigmasq * theta_in.T.dot(subSighat).dot(theta_in)/ theta_in.dot(subB).dot(theta_in) *  t[I0[1]]
            RQ_prop = 1/ sigmasq * thetahat_prop.T.dot(subSighat).dot(thetahat_prop)/ thetahat_prop.dot(subB).dot(thetahat_prop) * t[I0[1]]
            dif = thetahat_prop - theta_in
            Acc = RQ_prop - RQ - (rho_1 /2) * (norm(thetahat_prop)**2 - norm(theta_in)**2) * t[I0[1]] -  (np.sum(
                (-dif - tau * gradient_computer(thetahat_prop, subSighat, subB, sigmasq, rho_1) * t[I0[1]])**2)- np.sum((dif - tau * gradient)**2))/4/tau
            Acc = min(1, np.exp(Acc))
        #     print(Acc)
            if np.random.uniform(size=1) <= Acc:
                theta_in = thetahat_prop.copy()
                RQ = RQ_prop.copy()
            theta0[1, index01] = theta_in.copy()
        mAcc = mAcc + 1 / (jj + 1) *(Acc - mAcc)

        ## update 10
        n3 = len(index10)
        if n3 == 0:
            Acc = mAcc
        else:
            Z = np.random.normal(size = (n3))
            theta0[1, index10] = Z * np.sqrt( 1/rho_0/t[I0[1]])

            tau = np.exp(loggam[I0[0]])
            subSighat = A[index10,:][:, index10]
            subB = B[index10,:][:, index10]
            theta_in = theta0[0, index10]
            gradient = gradient_computer(theta_in, subSighat, subB, sigmasq, rho_1) * t[I0[0]]
            thetahat_prop = theta_in + tau* gradient + np.sqrt(2* tau ) * Z
            RQ = 1 / sigmasq * theta_in.T.dot(subSighat).dot(theta_in)/ theta_in.dot(subB).dot(theta_in) *t[I0[0]]
            RQ_prop = 1/ sigmasq * thetahat_prop.T.dot(subSighat).dot(thetahat_prop)/ thetahat_prop.dot(subB).dot(thetahat_prop) *t[I0[0]]
            dif = thetahat_prop - theta_in
            Acc = RQ_prop - RQ - (rho_1 /2) * (norm(thetahat_prop)**2 - norm(theta_in)**2) * t[I0[0]] -  (np.sum(
                (-dif - tau * gradient_computer(thetahat_prop, subSighat, subB, sigmasq, rho_1) * t[I0[0]])**2)- np.sum((dif - tau * gradient)**2))/4/tau
            Acc = min(1, np.exp(Acc))
        #     print(Acc)
            if np.random.uniform(size=1) <= Acc:
                thetain = thetahat_prop.copy()
    #             RQ = RQ_prop.copy()
            theta0[0, index10] = theta_in.copy()
        mAcc = mAcc+ 1 / (jj + 1) *(Acc - mAcc)  


        ## Update 11
        ## Sample proposals by maximal coupling
        n4 = len(index11)
        if n4 != 0:
            tau = np.ones(2)
            tau[0] = np.exp(loggam[I0[0]])
            tau[1] = np.exp(loggam[I0[1]])
            subSighat = A[index11,:][:, index11]
            subB = B[index11,:][:, index11]
            X_d = np.random.normal(size = (n4))
            X_prop = X_d * np.sqrt(2*tau[0]) + theta0[0, index11] + tau[0] * gradient_computer(theta0[0, index11], subSighat, subB, sigmasq, rho_1)
            W = np.random.uniform(size=1)

            if stats.multivariate_normal.pdf(X_d, np.zeros(n4)) * W <= stats.multivariate_normal.pdf(X_prop, theta0[0, index11] + tau[0] * gradient_computer(theta0[0, index11], subSighat, subB, sigmasq, rho_1), 2*tau[0]*np.identity(n4)) :
                Y_prop = X_prop.copy()
            else:
                Y_prop= np.random.normal(size = (n4)) * np.sqrt(2*tau[1]) + theta0[1, index11] + tau[1] * gradient_computer(theta0[1, index11], subSighat, subB, sigmasq, rho_1)
            U = np.random.uniform(size=1)

            theta_in = theta0[0,index11].copy()
            gradient = gradient_computer(theta_in, subSighat, subB, sigmasq, rho_1) * t[I0[0]]
            thetahat_prop = X_prop.copy()
            if theta_in.dot(subB).dot(theta_in) == 0:
                RQ = -float('inf')
            else:
                RQ = 1 / sigmasq * theta_in.T.dot(subSighat).dot(theta_in)/ theta_in.dot(subB).dot(theta_in) *t[I0[0]]
            if thetahat_prop.dot(subB).dot(thetahat_prop) != 0:
                RQ_prop = 1/ sigmasq * thetahat_prop.T.dot(subSighat).dot(thetahat_prop)/ thetahat_prop.dot(subB).dot(thetahat_prop) *t[I0[0]]
                dif = thetahat_prop - theta_in
                Acc = RQ_prop - RQ - (rho_1 /2) * (norm(thetahat_prop)**2 - norm(theta_in)**2) * t[I0[0]] -  (np.sum(
                    (-dif - tau[0] * gradient_computer(thetahat_prop, subSighat, subB, sigmasq, rho_1) * t[I0[0]])**2)- np.sum((dif - tau[0] * gradient)**2))/4/tau[0]
                Acc = min(1, np.exp(Acc))
                if U <= Acc:
                    theta0[0, index11] = thetahat_prop.copy()

            if update == True:
                loggam[I0[0]] = loggam[I0[0]] + 1/(jj**0.6 + 1) * (Acc - 0.3)


            theta_in = theta0[1, index11]
            gradient = gradient_computer(theta_in, subSighat, subB, sigmasq, rho_1) * t[I0[1]]
            thetahat_prop = Y_prop.copy()
            if theta_in.dot(subB).dot(theta_in) == 0:
                RQ = -float('inf')
            else:
                RQ = 1 / sigmasq * theta_in.T.dot(subSighat).dot(theta_in)/ theta_in.dot(subB).dot(theta_in) * t[I0[1]]
            if thetahat_prop.dot(subB).dot(thetahat_prop) != 0:
                RQ_prop = 1/ sigmasq * thetahat_prop.T.dot(subSighat).dot(thetahat_prop)/ thetahat_prop.dot(subB).dot(thetahat_prop) *t[I0[1]]
                dif = thetahat_prop - theta_in
                Acc = RQ_prop - RQ - (rho_1 /2) * (norm(thetahat_prop)**2 - norm(theta_in)**2) * t[I0[1]] -  (np.sum(
                    (-dif - tau[1] * gradient_computer(thetahat_prop, subSighat, subB, sigmasq, rho_1) * t[I0[1]])**2)- np.sum((dif - tau[1] * gradient)**2))/4/tau[1]
                Acc = min(1, np.exp(Acc))
                if U <= Acc:
                    theta0[1, index11] = thetahat_prop.copy()
            if update == True:
                loggam[I0[1]] = loggam[I0[1]] + 1/(jj**0.6 + 1) * (Acc - 0.3)



        ##  Swap temp
        ##Update temperature index
        W = np.random.uniform(size=1)
        U = np.random.uniform(size=1)
        I_prop = [0, 0]
        if I0[0] == 0:
            I_prop[0] = 1
            Acc = (loglik(A, B, sigmasq,  rho_0, rho_1, theta0[0], delta0[0], q) *(t[I_prop[0]]-  t[I0[0]]))- phi[0,I_prop[0]] + phi[0,I0[0]]
            Acc = min(1, np.exp(Acc)*0.5)
        elif I0[0] == ndim - 1:
            I_prop[0] = ndim - 2
            Acc = (loglik(A, B, sigmasq,  rho_0, rho_1, theta0[0], delta0[0], q) *(t[I_prop[0]]-  t[I0[0]]))- phi[0,I_prop[0]] + phi[0,I0[0]]
            Acc = min(1, np.exp(Acc)*0.5)
        else:
            if W<=0.5:
                I_prop[0] = I0[0] - 1
            else:
                I_prop[0] = I0[0] + 1
            Acc = (loglik(A, B, sigmasq,  rho_0, rho_1, theta0[0], delta0[0], q) *(t[I_prop[0]]-  t[I0[0]]))- phi[0,I_prop[0]] + phi[0,I0[0]]
            if I_prop[0] == 0:
                Acc = min(1, np.exp(Acc)*2)
            elif I_prop[0] == ndim-1:
                Acc = min(1, np.exp(Acc)*2)
            else:
                Acc = min(1, np.exp(Acc))
        if U <= Acc:
            I0[0] = I_prop[0]
        I[0,jj+1] = I0[0]


        if I0[1] == 0:
            I_prop[1] = 1
            Acc = (loglik(A, B, sigmasq,  rho_0, rho_1, theta0[1], delta0[1], q) *(t[I_prop[1]]-  t[I0[1]]))- phi[1, I_prop[1]] + phi[1, I0[1]]
            Acc = min(1, np.exp(Acc)*0.5)
        elif I0[1] == ndim - 1:
            I_prop[1] = ndim - 2
            Acc = (loglik(A, B, sigmasq,  rho_0, rho_1, theta0[1], delta0[1], q) *(t[I_prop[1]]-  t[I0[1]]))- phi[1, I_prop[1]] + phi[1, I0[1]]
            Acc = min(1, np.exp(Acc)*0.5)
        else:
            if W<=0.5:
                I_prop[1] = I0[1] - 1
            else:
                I_prop[1] = I0[1] + 1
            Acc = (loglik(A, B, sigmasq,  rho_0, rho_1, theta0[1], delta0[1], q) *(t[I_prop[1]]-  t[I0[1]]))- phi[1, I_prop[1]] + phi[1, I0[1]]
            if I_prop[1] == 0:
                Acc = min(1, np.exp(Acc)*2)
            elif I_prop[1] == ndim-1:
                Acc = min(1, np.exp(Acc)*2)
            else:
                Acc = min(1, np.exp(Acc))
        if U <= Acc:
            I0[1] = I_prop[1]
        I[1,jj+1] = I0[1]

        phi[:,I0] = phi[:,I0]+ aa
        visits[:, I0] = visits[:, I0]+1
        if np.max(np.abs(visits[0]/(jj+1) - 1/ ndim)) < 0.4 / ndim:
            aa[0] = aa[0]/2
            # print(jj, visits, "time when aa changes")
            visits[0] = np.zeros(ndim)

        if np.max(np.abs(visits[1]/(jj+1) - 1/ ndim)) < 0.4 / ndim:
            aa[1] = aa[1]/2
            # print(jj, visits, "time when aa changes")
            visits[1] = np.zeros(ndim)

        ## Record values
        thetaout =  theta0[0,:]*(delta0[0,:]) 
        normcons = np.sqrt(thetaout.dot(B).dot(thetaout))
        thetaout = thetaout /  normcons
    #         error[jj, 0] = norm(np.outer(thetaout, thetaout)  - proj_star1, 'fro')
    #         print(thetaout.T.dot(A).dot(thetaout) / n / 2 / thetaout.dot(B).dot(thetaout))
    #         print(thetaout[:p//2].dot(B[:p//2,:p//2]).dot(thetaout[:p//2]))
        error[0,jj, 1] = thetaout.T.dot(A).dot(thetaout) / n / 2
        norm1 = norm(thetaout[:p//2])
        norm2 = norm(thetaout[p//2:])
        error[0, jj, 2] = min((norm(thetaout[:p//2] / norm1 - vx[:,0])**2), (norm(thetaout[:p//2]/ norm1 + vx[:,0])**2))
        error[0, jj, 3] = min((norm(thetaout[p//2:] / norm2 - vy[:,0])**2), (norm(thetaout[p//2:] / norm2+ vy[:,0])**2))	#         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
    #         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
        Res_theta[0,jj, :] = theta0[0, :] / normcons
        Res_delta[0,jj, :] = delta0[0, :]

        thetaout =  theta0[1,:]*(delta0[1,:]) 
        normcons = np.sqrt(thetaout.dot(B).dot(thetaout))
        thetaout = thetaout /  normcons
    #         error[jj, 0] = norm(np.outer(thetaout, thetaout)  - proj_star1, 'fro')
    #         print(thetaout.T.dot(A).dot(thetaout) / n / 2 / thetaout.dot(B).dot(thetaout))
    #         print(thetaout[:p//2].dot(B[:p//2,:p//2]).dot(thetaout[:p//2]))
        error[1,jj-lag, 1] = thetaout.T.dot(A).dot(thetaout) / n / 2
        norm1 = norm(thetaout[:p//2])
        norm2 = norm(thetaout[p//2:])
        error[1,jj-lag, 2] = min((norm(thetaout[:p//2] / norm1 - vx[:,0])**2), (norm(thetaout[:p//2]/ norm1 + vx[:,0])**2))
        error[1,jj-lag, 3] = min((norm(thetaout[p//2:] / norm2 - vy[:,0])**2), (norm(thetaout[p//2:] / norm2+ vy[:,0])**2))	#         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
    #         print(thetaout.T.dot(A).dot(thetaout) / n / 2/ (thetaout.dot(B).dot(thetaout)))
        Res_theta[1,jj-lag, :] = theta0[1, :] / normcons
        Res_delta[1,jj-lag, :] = delta0[1, :]

        difference[jj] = norm(Res_theta[1,jj-lag, :] - Res_theta[0,jj, :] ) + np.sum(np.abs(Res_delta[1,jj-lag, :]- Res_delta[0,jj, :]))
        print(jj, I0, difference[jj])
        if np.sum(difference[jj]) < 0.0001:
            Niter = jj
            return error[:,:jj,:], Res_theta[:,:jj,:], Res_delta[:,:jj,:], swap_atem[:,:jj], swap[:,:jj], mAcc[:jj], loggam, Niter
    return error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam, Niter



