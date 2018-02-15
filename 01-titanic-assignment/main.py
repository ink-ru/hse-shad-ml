#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import division

import os, sys, re
import numpy as np
import pandas

'''
Более подробно со списком методов датафрейма можно познакомиться в документации - 
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
'''

csv_data = './ttnc.csv' # https://www.kaggle.com/c/titanic/data

data = pandas.read_csv(csv_data, index_col='PassengerId') # колонка PassengerId задает нумерацию строк данного датафрейма

def cls():
	os.system('cls' if os.name=='nt' else 'clear')

cls()

rows, columns = os.popen('stty size', 'r').read().split()

print ('▓'*int(columns))
# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

q = '1. Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.'
print(q)

# print(data.loc[data['Sex'].isin(['female'])].count())
# print(data[data.Sex == 'female'].count())
# print(data.Sex.str.contains(r'^\s*male\s*$').count())
# print(data.Sex.str.contains(r'^female').count())
# print(data.Sex.str.count('female'))

lst = data['Sex'].value_counts()
print(str(lst[0])+' '+str(lst[1]))

print ('═'*int(columns))
# ═════════════════════════════════════════════════════════════════════════

q = '2. Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров. Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.'
print(q)

survived = data.groupby('Survived').size()[1]
# survived = data.Survived.astype(bool).sum(axis=0)
total = len(data)
# print(total)
# print(survived)
print(int(round(survived/(total/100))))

print ('═'*int(columns))
# ═════════════════════════════════════════════════════════════════════════

q = '3. Какую долю пассажиры первого класса составляли среди всех пассажиров? Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.'
print(q)

fclass = data.groupby('Pclass').size()[1]
# print(fclass)
# print(data.Pclass[data.Pclass == 1].count())
# print(data.Pclass.count())
# print(int(round(fclass/(total/100))))
print(round(fclass/total*100, 2))

print ('═'*int(columns))
# ═════════════════════════════════════════════════════════════════════════

q = '4. Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров. В качестве ответа приведите два числа через пробел.'
print(q)

# avg = data.Age.sum()/data.Age.count()
mean = str(round(data.Age.mean(), 3))
median = str(data.Age.median())
print(mean+' '+median)

print ('═'*int(columns))
# ═════════════════════════════════════════════════════════════════════════

q = '5. Коррелируют ли число братьев/сестер/супругов с числом родителей/детей? Посчитайте корреляцию Пирсона между признаками SibSp и Parch.'
print(q)

print( round(data[['SibSp','Parch']].corr().Parch[0], 2))

print ('═'*int(columns))
# ═════════════════════════════════════════════════════════════════════════

q = '6. Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name).'
print(q)

names = data[data.Sex == 'female'].Name.values.tolist()

only_names = []
# names_data = pandas.DataFrame(columns=list('name'))

for n in names:
	n = n.replace('"', '')
	if n.find('(') > -1:
		n = re.sub(r'[^\(]+\(([A-z]+)(\s|[^)])*\).*', r'\1', n)
	else:
		n = re.sub(r'^[^,]+,\s+[^\.]+\.\s+([A-z]+).*', r'\1', n)
	only_names.append(n)
	# names_data.append(pandas.DataFrame([n], columns=list('name')))

data = pandas.DataFrame(data={'name':only_names}, index=range(len(only_names)))

# data = data.reset_index(drop=True)
# data.set_index('name')

data['freq'] = data.groupby('name')['name'].transform('count')

# with pandas.option_context('display.max_rows', None, 'display.max_columns', 3):
#     print(data)

print(str(data.loc[data['freq'].idxmax()])+', Anna')



print ('▓'*int(columns))

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

sys.exit(os.EX_OK) # code 0, all ok
