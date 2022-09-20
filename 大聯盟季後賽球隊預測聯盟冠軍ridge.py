# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 15:35:04 2021

@author: 呂星樺
"""


import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.linear_model import Ridge


from sklearn.linear_model import Lasso


from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeClassifier

from sklearn.linear_model import RidgeClassifierCV

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import auc, confusion_matrix,  f1_score, precision_score, recall_score, roc_curve


mlbteamdata = pd.read_csv('D:/ecomometric data in r\MLB static/1980-2019 adjust.ONLY.csv')

mlbteamdata_describe = mlbteamdata.describe()

mlbteamdata_describe_table = pd.DataFrame(mlbteamdata_describe)

mlbteamdata.columns

x_vars = [
    'nFld', 'CG.', 'Inn', 'PO', 'A', 'E', 'DP',
       'Rtot', 'nBat', 'BatAge', 'R/G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR',
       'RBI', 'SB', 'CS', 'BB', 'SO..', 'BA', 'OBP', 'SLG', 'OPS', 'GDP',
       'HBP', 'SH', 'SF', 'IBB', 'LOB', 'nP', 'PAge', 'RA/GP', 'ERA', 'GF',
       'CG', 'tSho', 'cSho', 'SV', 'IP', 'H.', 'RD', 'ER', 'HRD', 'BBD',
       'IBBD', 'SO', 'HBP.1', 'BK', 'WP', 'BF', 'WHIP', 'LOB..'
]





X = mlbteamdata.iloc[:,3:58]

y = mlbteamdata['chnu']

sc_x = StandardScaler()
X_std = sc_x.fit_transform(X)




X_train, X_test,y_train, y_test = train_test_split(X_std,y ,test_size = 0.33, random_state = 0)


## RidgeClassifier ##

for a in [0, 1, 10, 100, 1000]:
    lr_rg =  RidgeClassifier(alpha=a)
    lr_rg.fit(X_train, y_train)

    y_train_rigpred = lr_rg.predict(X_train)
    y_test_rigpred = lr_rg.predict(X_test)

    print('\n[Alpha = %d]' % a )
    print('MSE train: %.2f, test: %.2f' % (
                    mean_squared_error(y_train, y_train_rigpred),
                    mean_squared_error(y_test, y_test_rigpred)))  




print('accuracy(tree):%f'% accuracy_score(y_test, y_test_rigpred))



tn, fp, fn, tp = confusion_matrix(y_test, y_test_rigpred, labels=[0,1]).ravel()
classifier = classification_report(y_test, y_test_rigpred)

fpr, tpr, thresholds = roc_curve(y_test,
                                 y_test_rigpred,
                                 pos_label=1)
print('AUC: %.2f' % auc(fpr, tpr))



##RidgeClassifierCV##

rig = RidgeClassifierCV( alphas=(0.1, 1, 10,100,500, 1000),cv = 10)


rig.fit(X_train , y_train)


rig.coef_
rig.alpha_

y_rigpred = rig.predict(X_test)


print('accuracy(tree):%f'% accuracy_score(y_test, y_rigpred))


tn, fp, fn, tp = confusion_matrix(y_test, y_rigpred, labels=[0,1]).ravel()
classifier = classification_report(y_test,y_rigpred)

fpr, tpr, thresholds = roc_curve(y_test,
                                 y_rigpred
                                 )

print('AUC: %.2f' % auc(fpr, tpr))




































