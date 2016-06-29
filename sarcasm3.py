# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:21:47 2016

@author: asharma
"""

## standard imports
import pandas as pd
import numpy  as np
from collections import Counter
import matplotlib.pyplot as plt

from scipy import interp

## sklearn imports
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, auc

## nltk imports
import nltk
from nltk.data import load

tagdict = load('help/tagsets/upenn_tagset.pickle')

## define the I/O files
trg_tst_file = 'dataset/reddit_training.csv'
eval_file    = 'dataset/reddit_test.csv'
pred_file    = 'dataset/reddit_test_labels.csv'

## load the data
raw_model_data_df = pd.read_csv(trg_tst_file, encoding='utf-8')
raw_eval_data_df  = pd.read_csv(eval_file, encoding='utf-8')
    
def makeBOWModel(documents, max_features = 32):
    vec = TfidfVectorizer(max_df=0.1, sublinear_tf=True, stop_words='english', 
                          lowercase=False)
    vec.fit(documents)
    return vec
    
def makeLenFeature(df, col):
    return np.array([map(lambda x: len(x), df[col])])
    
def makeExpFeature(df, col, sp):
    return np.array([map(lambda x: int(sp in x), df[col])])

def makeQutoeFeature(df, col):
    return makeExpFeature(df, col, '"')

def makeExclamationFeature(df, col):
    return makeExpFeature(df, col, '!')
    
def makePOSFeature(df, col):
    fn = lambda y: map(lambda x: x[1], nltk.pos_tag(y.split()))
    lines_tags = map(fn, df[col])
    lines_tag_counts = map(lambda x: Counter(x), lines_tags)
    r = []
    for line_tag_counts in lines_tag_counts:
        m = dict(map(lambda x: (x,0), tagdict.keys()))
        for x in tagdict.keys():
            if line_tag_counts.get(x):
                m[x] = line_tag_counts[x]
        r.append(m.values())
    r_m = np.array(r)
    polyf = PolynomialFeatures(2, interaction_only=True, include_bias=False)
    return polyf.fit_transform(r_m)
    
def makeBOWFeatures(model, df, col):
    return model.transform(map(lambda x: x, df[col]))
    
def makeFeatures(bowModel, df, col):
    F  = makeBOWFeatures(bowModel, df, col).todense()
    f1 = makeExclamationFeature(df, col).T
    f2 = makeQutoeFeature(df, col).T
    f3 = makeLenFeature(df, col).T
    f4 = makePOSFeature(df, col)
    F1 = np.append(F, f1, axis=1) 
    F2 = np.append(F1, f2, axis=1)
    F3 = np.append(F2, f3, axis=1)
    F4 = np.append(F3, f4, axis=1)
    return F4

def score(y, p, t):
    return precision_recall_fscore_support(y, p >= t, average='binary', pos_label=1)
        
def estimateThreshold(clf, X, y):
    cv     = StratifiedKFold(y, n_folds=10)
    thresh = (np.array(range(10)) + 1.0) / 10
    threshs= []
    for i, (train, test) in enumerate(cv):
        pred_ = clf.fit(X[train], y[train]).predict_proba(X[test])[: ,1]
        perfs = np.array(map(lambda x: score(y[test], pred_, x)[2], thresh))
        idx   = np.argmax(perfs)
        threshs.append(thresh[idx])
    data   = Counter(threshs)
    return data.most_common(1)[0][0]

## pre-process and embed text in feature space
bowModel      = makeBOWModel(map(lambda x: x, raw_model_data_df['body']))
X             = makeFeatures(bowModel, raw_model_data_df, 'body')
ft            = lambda x: int(x[0] == 'yes')
y             = np.array(raw_model_data_df[['sarcasm_tag']].apply(ft, axis=1))
scaler        = MinMaxScaler(feature_range=(-1, 1))
X             = scaler.fit_transform(X)

## estimate threshold
estimator     = LogisticRegression(fit_intercept=False, C=1e9)
thresh        = estimateThreshold(estimator, X, y)
print 'Estimated probability threshold: ', thresh

## demonstrate predictive model ROC variance with threshold
if True:
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(y, n_folds=5)
    
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    
    for i, (train, test) in enumerate(cv):
        probas_ = estimator.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='Fold %d (auc = %0.2f)' % (i, roc_auc))
    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('roc.eps')
    plt.show()

## estimate performance
f_scores            = []
for train, test in StratifiedShuffleSplit(y, 10, test_size=0.3, random_state=0):
    f_scores.append(score(y[test], estimator.fit(X[train], y[train]).predict_proba(X[test])[: ,1], thresh))
f_scores_df         = pd.DataFrame(f_scores)
f_scores_df.columns = ['precision', 'recall', 'f-score', 'true_sum']
print 'Estimated prediction performance:\n', f_scores_df.describe() 

## prepare prediction model and run evaluation
estimator  = LogisticRegression(fit_intercept=False, C=1e9)
E          = makeFeatures(bowModel, raw_eval_data_df, 'body')
E          = scaler.transform(E)
predicted  = map(lambda x: 1 if x >= thresh else 0, estimator.fit(X, y).predict_proba(E)[:, 1])
print 'Evaluation set size and sarcasm counts: ', len(predicted), sum(predicted)

## write out predicted labels file
out_df          = raw_eval_data_df[['index']]
out_df['label'] = pd.Series(map(lambda x: "yes" if x == 1 else "no", predicted))
out_df.to_csv(pred_file, index=False)

