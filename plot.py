import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 

import seaborn as sns
sns.set()

def get_mean_std(ress):
    return np.mean(ress, axis=0), np.std(ress, axis =0)


if __name__ == '__main__':    
    sns.set_style("whitegrid")
    x = range(2000)
    #plt.figure(figsize=(10, 6))
    
    plt.figure(figsize=(8, 6))
    
    '''
    val = np.load('./results/50/bordaexp3_cnsmp_5_art.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='green',label = 'Vigilant D-EXP3')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='green', alpha=0.2)
    
    val = np.load('./results/50/bordaexp3_5_art.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='steelblue', label = 'D-EXP3')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='steelblue', alpha=0.2)
    
    val = np.load('./results/50/bordaTS_5_art.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='red', label = 'D-TS')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='red', alpha=0.2)'''
    
    
    
    val = np.load('./results/bordaexp3_2_cnsmp_5_art.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='green',label = 'Vigilant D-EXP3')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='green', alpha=0.2)
    
    val = np.load('./results/bordaexp3_2_5_art.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='steelblue', label = 'D-EXP3')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='steelblue', alpha=0.2)
    
    val = np.load('./results/bordaTS_2_5_art.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='red', label = 'D-TS')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='red', alpha=0.2)
    
    
    '''
    val = np.load('./results/50/bordaexp3_cnsmp_3_5_art.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='green',label = 'Vigilant D-EXP3')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='green', alpha=0.2)
    
    val = np.load('./results/50/bordaexp3_3_5_art.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='steelblue', label = 'D-EXP3')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='steelblue', alpha=0.2)
    
    val = np.load('./results/50/bordaTS_3_5_art.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='red', label = 'D-TS')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='red', alpha=0.2)'''

    
    
    
    
    ################################################################
    '''
    val = np.load('./results/bordaexp3_cnsmp_cars10.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='green',label = 'Vigilant D-EXP3')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='green', alpha=0.2)
    
    val = np.load('./results/bordaexp3_cars10.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='steelblue', label = 'D-EXP3')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='steelblue', alpha=0.2)
    
    val = np.load('./results/bordaTS_cars10.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='red', label = 'D-TS')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='red', alpha=0.2)'''
    
    
    '''
    val = np.load('./results/bordaexp3_2_cnsmp_cars10.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='green',label = 'Vigilant D-EXP3')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='green', alpha=0.2)
    
    val = np.load('./results/bordaexp3_2_cars10.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='steelblue', label = 'D-EXP3')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='steelblue', alpha=0.2)
    
    val = np.load('./results/bordaTS_2_cars10.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='red', label = 'D-TS')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='red', alpha=0.2)'''
    
    
    '''
    val = np.load('./results/bordaexp3_3_cnsmp_cars10.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='green',label = 'Vigilant D-EXP3')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='green', alpha=0.2)
    
    val = np.load('./results/bordaexp3_3_cars10.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='steelblue', label = 'D-EXP3')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='steelblue', alpha=0.2)
    
    val = np.load('./results/bordaTS_3_cars10.npy')
    val_mean, val_std = get_mean_std(val)
    plt.plot(x, val_mean,linewidth=2.0, color='red', label = 'D-TS')
    plt.fill_between(x, val_mean-val_std, val_mean+val_std, facecolor='red', alpha=0.2)'''
    
    
    
    
    '''ee = np.load('./results/eenet_results_{}.npy'.format(d))
    ee_mean, ee_std = get_mean_std(ee)
    plt.plot(x, ee_mean,linewidth=2.0, label = 'EE-Net')
    plt.fill_between(x, ee_mean-ee_std, ee_mean+ee_std, facecolor='red', alpha=0.2)'''


    plt.xlabel('Rounds',fontsize=20)
    plt.ylabel('Cumulative Reward',fontsize=20)
    plt.legend(prop={"size":25},loc='upper left')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.savefig('./figures/regret_5_art_1.pdf', dpi=500,bbox_inches='tight')
    plt.savefig('./figures/regret_5_art_2.pdf', dpi=500,bbox_inches='tight')
    #plt.savefig('./figures/regret_5_art_3.pdf', dpi=500,bbox_inches='tight')
    
    #plt.savefig('./figures/regret_cars10_1.pdf', dpi=500,bbox_inches='tight')
    #plt.savefig('./figures/regret_cars10_2.pdf', dpi=500,bbox_inches='tight')
    #plt.savefig('./figures/regret_cars10_3.pdf', dpi=500,bbox_inches='tight')
