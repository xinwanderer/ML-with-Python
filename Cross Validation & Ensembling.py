# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 22:23:46 2021

@author: 呂星樺
"""

import os, itertools, csv


import numpy as np

# pandas  0.25.1
import pandas as pd

# scikit-learn  0.21.3
from sklearn import datasets
load_iris = datasets.load_iris
make_moons = datasets.make_moons
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

# matplotlib  3.1.1
import matplotlib.pyplot as plt




iris = load_iris()
X, y = iris.data[:,[1,2]], iris.target

# hold out testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# hold out validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)



best_k, best_score = 0, 0
clfs = {}

for k in [1, 15 ,50]:                                                                      
    pipe = Pipeline([['sc',StandardScaler()],['clf', KNeighborsClassifier(n_neighbors = k)]])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    score = accuracy_score(y_val , y_pred)
    print('[{}-NN]\n Vlaidation accuracy: {}'.format(k,score))
    if score > best_score:
        best_k , best_score = k, score
    clfs[k] = pipe

y_pred = clfs[best_k].predict(X_test)
print('\n Test accuracy: %.2f (n_neighbors = %d selected by the holdout method)'%
      (accuracy_score(y_pred, y_test), best_k))




y_pred= clfs[15].predict(X_test)
print('Test accuracy: %.2f (n_neighbors=15 selected manually)' % 
      accuracy_score(y_test, y_pred))









iris = load_iris()
X, y = iris.data[:,[1,2]], iris.target

# hold out testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)



best_k, best_score = 0,0
for k in [1,15,50]:
    pipe = Pipeline([['sc', StandardScaler()],['clf', KNeighborsClassifier(n_neighbors = k)]])
    pipe.fit(X_train, y_train)
    
    scores = cross_val_score(pipe, X_train, y_train, cv = 5)
    print('[%d-NN]\n Validation accuracy: %.3f%s' % (k , scores.mean(), scores))
    
    if scores.mean()>best_score:
        best_k, best_score = k, scores.mean()
    clfs[k] = pipe



best_clf = clfs[best_k]
best_clf.fit(X_train, y_train)

y_pred = best_clf.predict(X_test)

print('Test accuracy: %.2f (n_neighbors=%d selected by 5-fold CV)' % 
      (accuracy_score(y_test, y_pred), best_k))









outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
inner_cv = KFold(n_splits=10, shuffle=True, random_state=1)




outer_scores = []

for i, (train_idx, test_idx) in enumerate(outer_cv.split(X,y)):
    print('[puter fold %d/5]'%(i+1))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    best_k , best_score = 0, 0
    clfs = {}

    for k in [1, 15, 50]:
        inner_scores = []
        
        for itrain_idx, val_idx in inner_cv.split(X_train, y_train):
            X_itrain, X_val = X_train[itrain_idx], X_train[val_idx]
            y_itrain, y_val = y_train[itrain_idx], y_train[val_idx]
            
            pipe = Pipeline([['sc', StandardScaler()],['clf', KNeighborsClassifier(n_neighbors = k)]])
            pipe.fit(X_itrain, y_itrain)
            
            y_pred = pipe.predict(X_val)
            inner_scores.append(accuracy_score(y_val, y_pred))
        score_mean = np.mean(inner_scores)
        if score_mean >best_score:
            best_k, best_score = k, score_mean
        clfs[k] = pipe
    best_clf = clfs[best_k]
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    outer_scores.append(accuracy_score(y_test,y_pred))
    print('Test accuracy: %.2f (n_neighbors = %d selected by inner 10-fold cv)'% 
          (outer_scores[i], best_k))

    
        


print('\nTest accuracy: %.2f (5x10 nested CV)' % np.mean(outer_scores))








outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
inner_cv = KFold(n_splits=10, shuffle=True, random_state=1)

outer_scores = []

for i, (train_idx , test_idx) in enumerate(outer_cv.split(X, y)):
    print('[outer fold %d/5]'% (i+1))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    pipe = Pipeline([['sc', StandardScaler()],['clf', KNeighborsClassifier()]])
    param_grid = {'clf__n_neighbors':[1,15,50]}
    
    gs = GridSearchCV(estimator = pipe , param_grid = param_grid,
                      scoring = 'accuracy', cv = inner_cv)
    gs.fit(X_train, y_train)
    best_clf = gs.best_estimator_
    best_clf.fit(X_train, y_train)    
    outer_scores.append(best_clf.score(X_test, y_test))
    print('Test accuracy: %.2f (n_neighbors=%d selected by inner 10-fold CV)' % 
                  (outer_scores[i], gs.best_params_['clf__n_neighbors']))

print('\nTest accuracy: %.2f (5x10 nested CV)' % np.mean(outer_scores))






def gen_data(num_data, sigma):
    x = 2* np.pi*(np.random.rand(num_data)-0.5)
    y = np.sin(x) + np.random.normal(0,sigma, num_data)
    return (x,y)




sigma = 1
n_range = range(10, 50, 2)
k_range = [5, 10]

poly = PolynomialFeatures(degree=2)
X = np.array([])
y = np.array([])
cv5_mean = []
cv5_std = []
cv10_mean = []
cv10_std = []
exp_mean = []
for n in n_range:
    # compute the bias and variance of cv5
    mse_test = []
    for i in range(500):
        x, y = gen_data(n, sigma)
        X = poly.fit_transform(x[:, np.newaxis])
        
        cv5 = KFold(n_splits=5, random_state=1,shuffle=True)
        for i, (train, test) in enumerate(cv5.split(X, y)):
            lr = LinearRegression()
            lr.fit(X[train], y[train])
            y_test_pred = lr.predict(X[test])
            mse_test.append(mean_squared_error(y[test], y_test_pred))
    
    cv5_mean.append(np.mean(mse_test))
    cv5_std.append(np.std(mse_test))
    
    # compute the bias and variance of cv10
    mse_test = []
    for i in range(500):
        x, y = gen_data(n, sigma)
        X = poly.fit_transform(x[:, np.newaxis])
        
        cv10 = KFold(n_splits=10, random_state=1,shuffle=True)
        for i, (train, test) in enumerate(cv10.split(X, y)):
            lr = LinearRegression()
            lr.fit(X[train], y[train])
            y_test_pred = lr.predict(X[test])
            mse_test.append(mean_squared_error(y[test], y_test_pred))
    
    cv10_mean.append(np.mean(mse_test))
    cv10_std.append(np.std(mse_test))
    
    # compute the expected generalization error of f_N
    mse_test = []
    for i in range(500):
        x, y = gen_data(n, sigma)
        X = poly.fit_transform(x[:, np.newaxis])
        lr = LinearRegression()
        lr.fit(X, y)
        x_test, y_test = gen_data(100, sigma)
        X_test = poly.transform(x_test[:, np.newaxis])
        y_test_pred = lr.predict(X_test)
        mse_test.append(mean_squared_error(y_test, y_test_pred))
    exp_mean.append(np.mean(mse_test))






plt.plot(n_range, cv5_mean, 
         markersize=5, label='5-Fold CV', color='blue')
plt.fill_between(n_range,
                 np.add(cv5_mean, cv5_std),
                 np.subtract(cv5_mean, cv5_std),
                 alpha=0.15, color='blue')

plt.plot(n_range, cv10_mean, 
         markersize=5, label='10-Fold CV', color='green')
plt.fill_between(n_range,
                 np.add(cv10_mean, cv10_std),
                 np.subtract(cv10_mean, cv10_std),
                 alpha=0.15, color='green')

plt.plot(n_range, exp_mean, 
         markersize=5, label='Exp', color='red')

plt.hlines(y=sigma, xmin=10, xmax=48, 
           label='Bayes', color='red', 
           linewidth=2, linestyle='--')

plt.legend(loc='upper right')
plt.xlim([10, 48])
plt.ylim([0, 5])
plt.xlabel('N')
plt.ylabel('MSE')








X, y = make_moons(n_samples=500, noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Class 0',
            c='r', marker='s', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1',
            c='b', marker='x', alpha=0.5)





pipe1 = Pipeline([['sc', StandardScaler()], ['clf', LogisticRegression(C = 10, random_state = 0, solver = "liblinear")]])
pipe2 = Pipeline([['clf', DecisionTreeClassifier(max_depth = 3, random_state = 0)]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', KNeighborsClassifier(n_neighbors = 5)]])





clf_labels = ['logisticregression','decisiontree','knn']

print('[individual]')

for pipe, label in zip([pipe1, pipe2, pipe3], clf_labels):
    scores = cross_val_score(estimator = pipe, X=X_train, y = y_train, cv = 10, scoring = 'roc_auc')
    print('%s : %.3f (+/- %.3f)'%(label, scores.mean(),scores.std()))






print('[voting]')
best_vt, best_w, best_score = None, (), 0

for a,b,c in list(itertools.permutations(range(0,3))):
    clf = VotingClassifier(estimators = [('lr', pipe1), ('dt', pipe2), ('knn', pipe3)],
                                         voting = 'soft', weights=[a,b,c])
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring = 'roc_auc')
    print('%s:%.3f (+/-%.3f)'%((a,b,c), scores.mean(), scores.std()))
    if best_score< scores.mean():
        best_vt, best_w, best_score = clf, (a,b,c), scores.mean()

print('\n best %s: %.3f'%(best_w , best_score))







clf_labels = ['logisticregression', 'decisiontree','knn','voting']
colors = ['black', 'orange','blue','green']
linestyles = ['-','-','-','--']

for clf, label, clr, ls in zip([pipe1, pipe2, pipe3, best_vt],clf_labels, colors,linestyles):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_true = y_test , y_score = y_pred)
    roc_auc = auc(x = fpr, y= tpr)
    
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc=%0.2f)' % (label, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([-0.02, 1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.xlabel('FPR')
    plt.ylabel('TPR')









tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = None, random_state = 0)

bag = BaggingClassifier(base_estimator = tree, n_estimators = 500,
                        max_samples = 0.7, bootstrap = True,max_features = 1,
                        bootstrap_features = False, n_jobs = 1, random_state = 1)


tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test , y_test_pred)

print('[decisiontree] accuracy-train = %.3f, accuracy_test = %.3f'%(tree_train, tree_test))


bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)


bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test,y_test_pred)
print('[Bagging] accuracy-train = %.3f, accuracy-test = %.3f' % (bag_train, bag_test))









tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
# single decision tree
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('[DecisionTree] accuracy-train = %.3f, accuracy-test = %.3f' % 
      (tree_train, tree_test))



ada = AdaBoostClassifier(base_estimator = tree , n_estimators = 500)
ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)


ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)

print('[AdaBoost] accuracy-train = %.3f, accuracy-test = %.3f' % 
      (ada_train, ada_test))





range_est = range(1,500)
ada_train, ada_test = [], []

for i in range_est:
    ada = AdaBoostClassifier(base_estimator = tree, n_estimators = i,
                             learning_rate = 1, random_state=1)
    
    ada = ada.fit(X_train, y_train)
    y_train_pred = ada.predict(X_train)
    y_test_pred = ada.predict(X_test)

    ada_train.append(accuracy_score(y_train, y_train_pred))
    ada_test.append(accuracy_score(y_test, y_test_pred))

      
plt.plot(range_est, ada_train, color='blue')
plt.plot(range_est, ada_test, color='green')
plt.xlabel('No. weak learners')
plt.ylabel('Accuracy')










ada16 = AdaBoostClassifier(base_estimator=tree, n_estimators=16)
ada16.fit(X_train, y_train)
y_train_pred = ada16.predict(X_train)
y_test_pred = ada16.predict(X_test)

ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)
print('[AdaBoost16] accuracy-train = %.3f, accuracy-test = %.3f' % 
      (ada_train, ada_test))





clf_labels = ['voting', 'bagging', 'adaboost16']

colors = ['orange','blue','green']

linestyles = ['-', '-', '-']



for clf, labels, clr, ls in zip([best_vt,bag,ada16], clf_labels, colors, linestyles):
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]

    fpr, tpr, thresholds = roc_curve(y_true = y_test, y_score = y_pred)
    roc_auc = auc(x = fpr, y = tpr)
    
    plt.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc=%0.2f)' % (label, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([-0.02, 0.6])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')












