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

os.chdir('/Users/apple/Dropbox/Sparse CCA/simulation/covid')
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

