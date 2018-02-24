#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import division
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import os, sys, re
import numpy as np
import pandas

'''
Подробнее о методе ближайших соседей - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
'''

def cls():
	os.system('cls' if os.name=='nt' else 'clear')

cls()

rows, columns = os.popen('stty size', 'r').read().split()

print ('▓'*int(columns))
# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

# 1. Загрузите выборку Wine по адресу https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
csv_data = './wine.csv'
df = pandas.read_csv(csv_data)

'''
2. Извлеките из данных признаки и классы. Класс записан в первом столбце (три варианта), признаки — в столбцах со второго по последний.
Более подробно о сути признаков можно прочитать по адресу https://archive.ics.uci.edu/ml/datasets/Wine
'''
t_df = df[0] # df.ix[:, 0]

# df = data_scale[:,1:14]
df = df.ix[:, list(range(1, 14))] # df.loc[:, 1:]

'''
3. Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold).
Создайте генератор разбиений, который перемешивает выборку перед формированием блоков (shuffle=True). 
Для воспроизводимости результата, создавайте генератор KFold с фиксированным параметром random_state=42.
В качестве меры качества используйте долю верных ответов (accuracy).
'''

kf = KFold(n_splits = 5, shuffle = True, random_state=42) # kf = KFold(len(t_df), n_folds=5, shuffle=True, random_state=42)

#acc = cross_val_score(KNeighborsClassifier(n_neighbors=k), df, t_df, cv=kf, scoring='accuracy')

'''
4. Найдите точность классификации на кросс-валидации для метода k ближайших соседей (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50.
При каком k получилось оптимальное качество? Чему оно равно (число в интервале от 0 до 1)? Данные результаты и будут ответами на вопросы 1 и 2.
'''

def test_accuracy(kf, X, y):
	scores = list()
	for n in range(1, 51):
		neighborsneighs = KNeighborsClassifier(n_neighbors=n) # KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs)
		# neigh.fit(df)
        scores.append(cross_val_score(neighborsneighs, X, y, cv=kf, scoring='accuracy'))

    return pandas.DataFrame(scores, k_range).mean(axis=1).sort_values(ascending=False)


accuracy = test_accuracy(kf, df, t_df)
top_accuracy = accuracy.head(1)

print(top_accuracy.index[0])
print(top_accuracy.values[0])

print ('═'*int(columns))
# ═════════════════════════════════════════════════════════════════════════

'''
5. Произведите масштабирование признаков с помощью функции sklearn.preprocessing.scale. Снова найдите оптимальное k на кросс-валидации.
'''

df = sklearn.preprocessing.scale(df)
accuracy = test_accuracy(kf, df, t_df)

# 6. Какое значение k получилось оптимальным после приведения признаков к одному масштабу?
# Приведите ответы на вопросы 3 и 4. Помогло ли масштабирование признаков?

top_accuracy = accuracy.head(1)
print(top_accuracy.index[0])
print(top_accuracy.values[0])


print ('▓'*int(columns))

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

sys.exit(os.EX_OK) # code 0, all ok
