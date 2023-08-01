import matplotlib.pyplot as plt
import numpy as np
def plotItr(error_dict,itr_max = 2000, figname = False):
    colors = ['r','b','g','c','k','m','y']
    markers = ['s','o','v','^','+','x','>']
    linestyles = ['-','--','-','-',':',':','-.']
    i = 0
    
    fig = plt.figure()
    for algo, error in error_dict.items():
        c = colors[i]
        mkr = markers[i]
        ls = linestyles[i]
        fmt = ls + c + mkr
        chosen_points = np.arange(0, itr_max, 100)
        markers_on = np.arange(0, itr_max//100, itr_max//500)
        mkrstyle = {'linewidth': 1, 'markersize': 3, 'markeredgewidth': 2, \
                       'markeredgecolor': c, 'markerfacecolor': 'None'}
        
        plt.plot(chosen_points, error[chosen_points], fmt, markevery=markers_on, **mkrstyle, label = algo)
        i = i + 1
    labelsize = 20
    ticksize = 16
    legendsize = 12
    plt.ylim(0,2.2)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel(r'Iteration',fontsize = labelsize)
    plt.ylabel(r'Squared l2 error',fontsize = labelsize)
    plt.gca().legend(fontsize=legendsize)
    plt.tight_layout()
    if figname:
        # print('Fig saved!')
        plt.savefig(figname+'.png',bbox_inches = 0)
    plt.show()
    
def plotTime(error_dict,itr_max = 2000, figname = False):
    colors = ['r','b','g','c','k','m','y']
    markers = ['s','o','v','^','+','x','>']
    linestyles = ['-','--','-','-',':',':','-.']
    i = 0
    
    fig = plt.figure()
    for algo, error in error_dict.items():
        c = colors[i]
        mkr = markers[i]
        ls = linestyles[i]
        fmt = ls + c + mkr
        chosen_points = np.arange(0, itr_max, 100)
        markers_on = np.arange(0, itr_max//100, itr_max//500)
        mkrstyle = {'linewidth': 1, 'markersize': 3, 'markeredgewidth': 2, \
                       'markeredgecolor': c, 'markerfacecolor': 'None'}
        
        plt.plot(error[0][chosen_points], error[1][chosen_points], fmt, markevery=markers_on, **mkrstyle, label = algo)
        i = i + 1
    labelsize = 20
    ticksize = 16
    legendsize = 12
    plt.ylim(0,2.2)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.xlabel(r'Computation time/s',fontsize = labelsize)
    plt.ylabel(r'Squared l2 error',fontsize = labelsize)
    plt.gca().legend(fontsize=legendsize)
    plt.tight_layout()
    if figname:
        # print('Fig saved!')
        plt.savefig(figname+'.png',bbox_inches = 0)
    plt.show()