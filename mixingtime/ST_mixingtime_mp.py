#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('..')
from SCCA import *
import pickle
import os
import sys
from multiprocessing import Pool

#re_ST = np.zeros(shape = (nd,nrun, 2))

nd = 50
#for index in range(50):
def outer_loop_func(index):
    nrun = 50
    re_ST_per = np.zeros(shape = (nrun,2))
    d = 100 * (index + 1)
    if d >100:
        size  = 100
    else:
        size = False

    print(d)
    n = int(d/2)
    s = 6
    lam = 0.9
    for run in range(50):
    # np.random.seed(100)
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


        A = np.vstack((np.hstack((np.zeros(shape = (d//2, d//2)), SigXY))
                            ,np.hstack((SigXY.T, np.zeros(shape = (d//2, d//2)))))) * n

        B = np.concatenate((np.concatenate((SigX, np.zeros(shape = (d//2, d//2))), axis=0),
                            np.concatenate((np.zeros(shape = (d//2, d//2)), SigY), axis=0)), axis = 1) /2


        # In[ ]:


        now1 = datetime.now()
        error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam, Niter = ST_mix(Ahat, Bhat, vx, vy, d, n, lag = d, temp=[1,0.9, 0.8,0.7]
                                                        , sigmasq = 10/n**0.5,initial=False, Niter = d*100+10000, update = True, loggam = np.zeros(4)-4, rho_0 = n/10, batch = size, truncate = False)



        # Saving the objects:
        # with open('result/ST_mix%i.pkl' %d, 'wb') as f:  # Python 3: open(..., 'wb')
        #     pickle.dump([d, n, [1,0.9, 0.8,0.7], True,[-3.94685961, -4.01251142, -4.93310664, -3.26939622],error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam, Niter], f)
        ## d, n, lag, temp =[1,0.9, 0.8,0.7], Update = True,loggam = [-3.94685961, -4.01251142, -4.93310664, -3.26939622], error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam, Niter

        re_ST_per[run,0] = Niter
        with open('result_sigmasq/mixingtime%i-%i.pkl' %(d,run), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(Niter, f)
    # with open('mixingtime%i.pkl' %d , 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump(re_ST_per, f)
    return re_ST_per
    # # Getting back the objects:
    # with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
    #     obj0, obj1, obj2 = pickle.load(f)


if __name__=='__main__':
#    orig_stdout = sys.stdout
#    ff = open('out1.txt', 'w')
#    sys.stdout = ff
    nslots=int(os.getenv('NSLOTS'))
    with Pool(nslots) as pool:
        re_ST = pool.map(outer_loop_func, range(0, 50))
    # with open('mixingtime.pkl' , 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump(re_ST, f)
    sys.stdout = orig_stdout
    ff.close()

















































