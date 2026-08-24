"""Evaluation: Pearson/Spearman, ECE, Brier, Ramachandran proxy."""
import torch, numpy as np
from sklearn.metrics import brier_score_loss
from scipy.stats import spearmanr, pearsonr

def brier(pred_prob, target_prob):
    # pred_prob 0-1, target 0-1
    return ((pred_prob - target_prob)**2).mean()

def ece_score(conf, acc, n_bins=10):
    conf = np.array(conf); acc = np.array(acc)
    ece=0.0
    for b in range(n_bins):
        low=b/n_bins; high=(b+1)/n_bins
        mask = (conf>=low) & (conf<high) if b<n_bins-1 else (conf>=low) & (conf<=high)
        if mask.sum()==0: continue
        ece+= abs(conf[mask].mean() - acc[mask].mean()) * mask.sum()/len(conf)
    return ece

if __name__=="__main__":
    # stub demo
    print("Evaluate CASP16/CAMEO: Pearson, Spearman, ECE, Brier, Ramachandran Z")
    # Example synthetic
    y_true=np.random.rand(100)*100
    y_pred=y_true + np.random.randn(100)*10
    conf=y_pred/100.0
    acc=y_true/100.0
    print("Pearson", pearsonr(y_pred,y_true)[0])
    print("Spearman", spearmanr(y_pred,y_true)[0])
    print("ECE", ece_score(conf, acc))
    print("Brier", brier(conf, acc))
