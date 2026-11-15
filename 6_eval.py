import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyriemann.classification import MDM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE

if __name__ == '__main__':
    xp = np.load('X_psd.npy')
    xc = np.load('X_cov.npy')
    xt = np.load('X_ts.npy')
    yl = np.load('y_labels.npy')
    gl = np.load('groups.npy')

    logo = LeaveOneGroupOut()

    mdls = {
        'TS-SVM': (Pipeline([('sc', StandardScaler()), ('svm', LinearSVC(class_weight='balanced', dual='auto', random_state=42))]), xt, 'decision_function'),
        'TS-LR': (Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(class_weight='balanced', max_iter=1000))]), xt, 'decision_function'),
        'MDM': (MDM(metric=dict(mean='riemann', distance='riemann')), xc, 'predict_proba'),
        'RF': (RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42), xp, 'predict_proba'),
        'Euc-LR': (Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(class_weight='balanced', max_iter=1000))]), xp, 'decision_function')
    }

    res = []
    for nm, (m, x, mth) in mdls.items():
        sc = cross_val_predict(m, x, yl, groups=gl, cv=logo, method=mth, n_jobs=-1)
        if mth == 'predict_proba':
            sc = sc[:, 1]
            pr = (sc >= 0.5).astype(int)
        else:
            pr = (sc >= 0.0).astype(int)

        res.append({
            'Model': nm,
            'Accuracy': accuracy_score(yl, pr),
            'ROC-AUC': roc_auc_score(yl, sc),
            'Precision': precision_score(yl, pr),
            'Recall': recall_score(yl, pr),
            'F1-Score': f1_score(yl, pr)
        })

    df_res = pd.DataFrame(res)
    print(df_res.to_string(index=False))
    df_res.to_csv('loso_metrics.csv', index=False)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(yl), size=min(2500, len(yl)), replace=False)

    xp_sub = StandardScaler().fit_transform(xp[idx])
    xt_sub = StandardScaler().fit_transform(xt[idx])
    yl_sub = yl[idx]

    tsne_euc = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1).fit_transform(xp_sub)
    tsne_rie = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1).fit_transform(xt_sub)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for lbl, nm, clr in [(0, 'Healthy Control (CN)', '#1f77b4'), (1, 'Alzheimer\'s (AD)', '#d62728')]:
        msk = (yl_sub == lbl)
        ax[0].scatter(tsne_euc[msk, 0], tsne_euc[msk, 1], c=clr, label=nm, alpha=0.6, s=15, edgecolors='none')
        ax[1].scatter(tsne_rie[msk, 0], tsne_rie[msk, 1], c=clr, label=nm, alpha=0.6, s=15, edgecolors='none')

    ax[0].set_title('Euclidean Feature Space (Welch PSD)')
    ax[0].set_xlabel('t-SNE Dimension 1')
    ax[0].set_ylabel('t-SNE Dimension 2')
    ax[0].legend()

    ax[1].set_title('Riemannian Tangent Space (SPD Manifold)')
    ax[1].set_xlabel('t-SNE Dimension 1')
    ax[1].set_ylabel('t-SNE Dimension 2')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('tsne_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
