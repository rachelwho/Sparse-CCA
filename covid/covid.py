#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:58:19 2020

@author: apple
"""
import pickle
import os

import sys

# os.getcwd()
# os.chdir('/Users/apple/Dropbox/Sparse CCA/simulation')
sys.path.append('..')
from SCCA import *

import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage


from scipy.spatial.distance import squareform

os.chdir(os.path.dirname(os.path.abspath("covid.py")))
data2 = pd.read_excel("../../Real data/Covid_data/mmc2.xlsx", 1, header = 1, index_col = 0)
data1 = pd.read_excel("../../Real data/Covid_data/mmc1.xlsx", 1, index_col = 2)
data3 = pd.read_excel("../../Real data/Covid_data/mmc2.xlsx", 2, header = 1, index_col = 0)

data_p = pd.concat([data2, data3], join = 'inner', axis = 1)
clin_f = [ 'Group d','Sex g', 'Age (year)',
        'BMI h', 'WBC count, ×109/L', 'Lymphocyte count, ×109/L',
        'Monocyte count, ×109/L', 'Platelet count, ×109/L',
        'CRP i, mg/L', 'ALT j,  U/L', 'AST k, U/L', 'GGT l, U/L',
        'TBIL m, μmol/L', 'DBIL n, μmol/L', 'Creatinine, μmol/L',
        'Glucose, mmol/L']
# clin_f = np.array([ 'Lymphocyte count, ×109/L',
#        'Monocyte count, ×109/L', 
#        'CRP i, mg/L', 'AST k, U/L'])
index1 = data1.index
index1 = index1[index1 != '/']
index_p = data_p.columns
index = list(set(index1).intersection(index_p))

clin = data1.loc[index][clin_f].fillna(0).values
clin[clin=='/'] = 0
clin[clin == '<1.3'] = 0
clin = np.array(list(clin), dtype = float)

data_new = data_p[index].dropna(thresh = 50)
protein = data_new.fillna(0).values.T
protein_f = data_new.index

np.savetxt('X_covid.csv', clin, delimiter = ",")
np.savetxt('Y_covid.csv', protein, delimiter = ",")
np.savetxt('group_covid.csv', clin[:,0], delimiter = ",")

p1 = len(clin_f)
p2 = len(protein_f)
n = len(index)
X = clin
Y = protein
# center columns of X

Rhat = pd.read_csv("Rhat.csv", header = 0, index_col= 0).values
Sighat_X = Rhat[:p1, :p1]
Sighat_Y = Rhat[p1:, p1:]
Sighat_XY = Rhat[:p1, p1:]
Ahat = np.vstack((np.hstack((np.zeros(shape = (p1, p1)), Sighat_XY))
                    ,np.hstack((Sighat_XY.T, np.zeros(shape = (p2, p2))))))

Bhat = np.vstack((np.hstack((Sighat_X, np.zeros(shape = (p1, p2)))),
                    np.hstack((np.zeros(shape = (p2, p1)), Sighat_Y)))) / (n*2)




# indd = np.where(Sighat_XY >0.8)
# index_X = np.delete(np.arange(p1), indd[0])
# p1_new = len(index_X)
# index_Y = np.delete(np.arange(p2), indd[1]) + p1
# p2_new = len(index_Y)
# Sighat_X = Rhat[index_X, :][:, index_X]
# Sighat_Y = Rhat[index_Y, :][:, index_Y]
# Sighat_XY = Rhat[index_X, :][: , index_Y]
# p1 = p1_new
# p2 = p2_new
# Ahat = np.vstack((np.hstack((np.zeros(shape = (p1_new, p1_new)), Sighat_XY))
#                     ,np.hstack((Sighat_XY.T, np.zeros(shape = (p2_new, p2_new))))))

# Bhat = np.vstack((np.hstack((Sighat_X, np.zeros(shape = (p1_new, p2_new)))),
#                     np.hstack((np.zeros(shape = (p2_new, p1_new)), Sighat_Y)))) / (n*2)

size = 100
now1 = datetime.now()
error, Res_theta, Res_delta, swap_atem, swap, mAcc, loggam, I, phi,aa, visits = Simu_var(Ahat, Bhat, p1,p2, n, temp=[1, 0.9, 0.8, 0.7], sigmasq = 1,initial=False, Niter = 100000, update = True, loggam = [-3.94685961, -4.01251142, -4.93310664, -3.26939622, -3, -3, -3, -3, -3, -3], batch = size)



with open('/Users/apple/Desktop/sCCA/res_covid_Rhat.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump((error, Res_theta, Res_delta, I),f)
with open('/Users/apple/Desktop/sCCA/res_covid_Rhat.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    error, Res_theta, Res_delta, I= pickle.load(f)

np.savetxt( "real_theta.txt", Res_theta[I==0,], fmt='%.4f')
np.savetxt( "real_delta.txt", Res_delta[I==0,], fmt='%d')
np.savetxt( "real_z.txt", I==0, fmt='%d')


ll = Res_delta[I==0].shape[0]

d = p1+p2
fig, ax = plt.subplots(4, 1, sharey=False, tight_layout=True , figsize = (10,10))
ax[0].plot(Res_theta[I==0, 10][0:ll]) 
ax[0].set_title(r'$\theta$')

ax[1].plot(Res_theta[I==0, 21][0:ll]) 
ax[1].set_title(r'$\theta$')

ax[2].plot(Res_theta[I==0, 488][0:ll]) 
ax[2].set_title(r'$\theta$')

## eigenvalue
ax[3].plot(error[I==0,1][0:ll], 'b.')
ax[3].set_title(r"eigenvalues")
ax[3].grid()

plt.boxplot(error[I==0,1][4000:ll])

plt.plot(np.sum(Res_delta[I==0][50:ll]==1, axis =1))

plt.plot(I[40:ll], 'r.')

clin_f[ind_X[0]]
Out[149]: 'CRP i, mg/L'

protein_f[ind_Y[0]-p1]
Out[150]: 'P02748'

ind = np.where(np.array(Res_delta[I==0][400:])==1)
plt.plot(ind[0], ind[1], 'r.')


num = 7
count = np.sum(Res_delta[I==0][400:], axis = 0)
ind = count.argsort()[-num:][::-1]
ind_X = ind[ind<p1]
ind_Y = ind[ind>=p1]

clin_f[ind_X[0]]
protein_f[ind_Y[0]-p1]
clin_f[ind_X]
protein_f[ind_Y-p1]


chosen_X = Res_delta[I==0][2000:, ind_X]
covmat_X = np.corrcoef(chosen_X.T)
dissimilarity_X = 1 - np.abs(covmat_X)
dissimilarity_X = np.round(dissimilarity_X, 2)  
n_X = len(ind_X)

chosen_Y = Res_delta[I==0][2000:, ind_Y]
np.sum(chosen_Y, axis = 0)
covmat_Y = np.corrcoef(chosen_Y.T)
dissimilarity_Y = 1 - np.abs(covmat_Y)
dissimilarity_Y = np.round(dissimilarity_Y, 2)  
n_Y = len(ind_Y)

for i in range(n_X):
    print(ind_X[i])
    for j in range(n_X):
        print('&',"{:.3f}".format(covmat_X[i,j]))
    print('\\\\')
for i in range(n_X):
    print('&',ind_X[i])


for i in range(n_Y):
    print(ind_Y[i])
    for j in range(n_Y):
        print('&',"{:.3f}".format(covmat_Y[i,j]))
    print('\\\\')

for i in range(n_Y):
    print('&',ind_Y[i])

# L_X = linkage(squareform(dissimilarity_X), method='average')
L_X = linkage(scipy.spatial.distance.pdist(chosen_X.T), "single")
L_X = linkage(covmat_X)
dendrogram(L_X, leaf_rotation=90, leaf_font_size=8, labels = cols[ind_X,0])
plt.xlabel("miRNA")
plt.ylabel("dissimilarities", )
plt.title("micro RNAs")
plt.savefig('../manuscript/biometrika2018/miRNA.pdf', bbox_inches='tight')
fcluster(L_X, 0.5)

L_Y = linkage(squareform(dissimilarity_Y), method='average')
L_Y = linkage(scipy.spatial.distance.pdist(chosen_Y.T), "single")
L_Y = linkage(covmat_Y)
dendrogram(L_Y, leaf_rotation=90, leaf_font_size=8, labels = protein_f[ind_Y-p1])
plt.xlabel("mRNA")
plt.ylabel("dissimilarities")
plt.title("message RNAs")
plt.savefig('../manuscript/biometrika2018/mRNA.pdf', bbox_inches='tight')
fcluster(L_Y, 0.7)



np.where(Res_delta[I==0,:][1000,:]==1)
np.where(Res_delta[I==0,:][2100,:]==1)
kRes_theta[I==0,:][2100,[130, 452, 923]]
thetaout = Res_theta[I==0,:][3000,]*Res_delta[I==0,:][3000,]
thetaout.dot(Bhat).dot(thetaout)
thetaout.T.dot(A).dot(thetaout) / n / 2
thetaout[:p1].dot(Sighat_X).dot(thetaout[:p1])/n
thetaout[:p1].dot(Sighat_XY).dot(thetaout[p1:])/np.sqrt(thetaout[:p1].dot(Sighat_X).dot(thetaout[:p1]))/np.sqrt(thetaout[p1:].dot(Sighat_Y).dot(thetaout[p1:]))
thetaout[:p1].dot(Sighat_XYt).dot(thetaout[p1:])/np.sqrt(thetaout[:p1].dot(Sighat_Xt).dot(thetaout[:p1]))/np.sqrt(thetaout[p1:].dot(Sighat_Yt).dot(thetaout[p1:]))







