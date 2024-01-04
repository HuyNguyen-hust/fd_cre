import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

plt.style.use('seaborn')
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12)

def smooth(y):
    x = np.array(range(len(y)))
    X_Y_Spline = make_interp_spline(x, y)
    X_ = np.linspace(x.min(), x.max(), 25)
    Y_ = X_Y_Spline(X_)
    
    return X_, Y_

def get_pca(path):
    cos = {}
    eigval = {}
    for res in os.listdir(path):
        p = os.path.join(path, res)
        with open(p) as f:
            if res.startswith('cos'):
                rel = res.split('cos_')[1]
                cos[rel] = [float(t) for t in f.read()[1:-1].split(',')]
            elif res.startswith('eigval'):
                rel = res.split('eigval_')[1]
                eigval[rel] = [float(t) for t in f.read()[1:-1].split(',')]
    
    return {'eigval': eigval, 'cos': cos}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz_path', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    
    models = ['ce']
    
    res = {}
    
    for model in models:
        res[model] = get_pca(os.path.join(args.viz_path, args.dataset, model))
    
    # eigval
    for rel in tqdm(res[models[0]]['eigval']):
        for model in models:
            tmp_eig = res[model]['eigval'][rel]
            sns.set_style("whitegrid")
            plt.plot(tmp_eig, label=model)
            plt.legend(loc="upper right")
        plt.title(f'{args.dataset}_{rel}', fontsize=15)
        plt.xlabel('Eigenvalue rank index', fontsize=15)
        plt.ylabel('Eigenvalues', fontsize=15)
        plt.savefig(os.path.join(os.path.join(args.viz_path, args.dataset), 'eigval_' + rel.replace(':', '_')) + '.pdf', format='pdf')
        plt.clf()
    
    # cos
    for rel in tqdm(res[models[0]]['cos']):
        for model in models:
            tmp_cos = res[model]['cos'][rel]
            sns.set_style("whitegrid")
            # x, y = smooth(tmp_cos)
            plt.plot(tmp_cos, label=model)
            plt.legend(loc="upper right")
        plt.title(f'{args.dataset}_{rel}', fontsize=15)
        plt.xlabel('Eigenvalue rank index', fontsize=15)
        plt.ylabel("Cosine values of correspoding angles", fontsize=15)
        plt.savefig(os.path.join(os.path.join(args.viz_path, args.dataset), 'cos_' + rel.replace(':', '_')) + '.pdf', format='pdf')
        plt.clf()

if __name__ == '__main__':
    main()