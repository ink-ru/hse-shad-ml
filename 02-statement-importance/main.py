#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import division
from sklearn.tree import DecisionTreeClassifier

import os, sys, re
import numpy as np
import pandas

'''
Подробнее про решающие деревья в sklearn - http://scikit-learn.org/stable/modules/tree.html
'''

def cls():
	os.system('cls' if os.name=='nt' else 'clear')

cls()

rows, columns = os.popen('stty size', 'r').read().split()

# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
csv_data = './ttnc.csv' # https://www.kaggle.com/c/titanic/data

data = pandas.read_csv(csv_data, index_col='PassengerId') # колонка PassengerId задает нумерацию строк данного датафрейма

# Преобразуем строковый признак Sex в бинарный

# data.Sex.apply(lambda x:  0 if x.find('male') > -1 else 1)
data['Sex'] = np.where(data['Sex'] == 'male' , 1, 0)

# Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.

data = data[np.isfinite(data['Age'])] # data1 = np.isnan(data)

# Выделите целевую переменную — она записана в столбце Survived
t_data = data['Survived'] # target data

# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).

col_list = ['Pclass', 'Fare', 'Age', 'Sex']
data = data[col_list] # df.filter(items=['Pclass', 'Fare', 'Age', 'Sex'])

# Обучите решающее дерево с параметром random_state=241

clf = DecisionTreeClassifier(random_state=241)
clf.fit(data, t_data)
importances = clf.feature_importances_

print ('▓'*int(columns))
# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

print(importances)

# print ('═'*int(columns))
# ═════════════════════════════════════════════════════════════════════════


print ('▓'*int(columns))

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

sys.exit(os.EX_OK) # code 0, all ok
