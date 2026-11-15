import numpy as np
import joblib
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score

xp = np.load('X_psd.npy')
xc = np.load('X_cov.npy')
yl = np.load('y_labels.npy')
gl = np.load('groups.npy')

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(gss.split(xc, yl, gl))

ts_svm = Pipeline([
    ('ts', TangentSpace(metric='riemann')),
    ('sc', StandardScaler()),
    ('svm', SVC(kernel='linear', C=0.01, class_weight='balanced'))
])

ts_lr = Pipeline([
    ('ts', TangentSpace(metric='riemann')),
    ('sc', StandardScaler()),
    ('lr', LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000))
])

mdm = MDM(metric=dict(mean='riemann', distance='riemann'))

rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)

euc_lr = Pipeline([
    ('sc', StandardScaler()),
    ('lr', LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000))
])

ts_svm.fit(xc[tr], yl[tr])
ts_lr.fit(xc[tr], yl[tr])
mdm.fit(xc[tr], yl[tr])
rf.fit(xp[tr], yl[tr])
euc_lr.fit(xp[tr], yl[tr])

print("TS-SVM:", accuracy_score(yl[te], ts_svm.predict(xc[te])), roc_auc_score(yl[te], ts_svm.decision_function(xc[te])))
print("TS-LR:", accuracy_score(yl[te], ts_lr.predict(xc[te])), roc_auc_score(yl[te], ts_lr.decision_function(xc[te])))
print("MDM:", accuracy_score(yl[te], mdm.predict(xc[te])), roc_auc_score(yl[te], mdm.predict_proba(xc[te])[:, 1]))
print("RF:", accuracy_score(yl[te], rf.predict(xp[te])), roc_auc_score(yl[te], rf.predict_proba(xp[te])[:, 1]))
print("Euc-LR:", accuracy_score(yl[te], euc_lr.predict(xp[te])), roc_auc_score(yl[te], euc_lr.decision_function(xp[te])))

xt = TangentSpace(metric='riemann').fit_transform(xc)
np.save('X_ts.npy', xt)

joblib.dump({
    'ts_svm': ts_svm,
    'ts_lr': ts_lr,
    'mdm': mdm,
    'rf': rf,
    'euc_lr': euc_lr
}, 'models.pkl')
