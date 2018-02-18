#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, os
import pandas
import sklearn.metrics as metrics

# 1. Загрузите файл classification.csv. В нем записаны истинные классы объектов выборки (колонка true) и ответы
# некоторого классификатора (колонка predicted).

df = pandas.read_csv('classification.csv')

# 2. Заполните таблицу ошибок классификации. Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям.
# Например, FP — это количество объектов, имеющих класс 0, но отнесенных алгоритмом к классу 1.
# Ответ в данном вопросе — четыре числа через пробел.

clf_table = {'tp': (1, 1), 'fp': (0, 1), 'fn': (1, 0), 'tn': (0, 0)}
for name, res in clf_table.items():
    clf_table[name] = len(df[(df['true'] == res[0]) & (df['pred'] == res[1])])

ans1 = '{tp} {fp} {fn} {tn}'.format(**clf_table)

# 3. Посчитайте основные метрики качества классификатора:

# Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
acc = metrics.accuracy_score(df['true'], df['pred'])

# Precision (точность) — sklearn.metrics.precision_score
pr = metrics.precision_score(df['true'], df['pred'])

# Recall (полнота) — sklearn.metrics.recall_score
rec = metrics.recall_score(df['true'], df['pred'])

# F-мера — sklearn.metrics.f1_score
f1 = metrics.f1_score(df['true'], df['pred'])

# В качестве ответа укажите эти четыре числа через пробел.
ans2 = '{:0.2f} {:0.2f} {:0.2f} {:0.2f}'.format(acc, pr, rec, f1)

# 4. Имеется четыре обученных классификатора. В файле scores.csv записаны истинные классы и значения степени
# принадлежности положительному классу для каждого классификатора на некоторой выборке:
# для логистической регрессии — вероятность положительного класса (колонка score_logreg),
# для SVM — отступ от разделяющей поверхности (колонка score_svm),
# для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
# для решающего дерева — доля положительных объектов в листе (колонка score_tree).
# Загрузите этот файл.

df2 = pandas.read_csv('scores.csv')

# 5. Посчитайте площадь под ROC-кривой для каждого классификатора. Какой классификатор имеет наибольшее значение
# метрики AUC-ROC (укажите название столбца с ответами этого классификатора)?
# Воспользуйтесь функцией sklearn.metrics.roc_auc_score.

scores = {}
for clf in df2.columns[1:]:
    scores[clf] = metrics.roc_auc_score(df2['true'], df2[clf])

ans3 = repr(pandas.Series(scores).sort_values(ascending=False).head(1).index[0])

# 6. Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
# Какое значение точности при этом получается? Чтобы получить ответ на этот вопрос, найдите все точки
# precision-recall-кривой с помощью функции sklearn.metrics.precision_recall_curve. Она возвращает три массива:
# precision, recall, thresholds. В них записаны точность и полнота при определенных порогах,указанных в массиве
# thresholds. Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.

pr_scores = {}
for clf in df2.columns[1:]:
    pr_curve = metrics.precision_recall_curve(df2['true'], df2[clf])
    pr_curve_df = pandas.DataFrame({'precision': pr_curve[0], 'recall': pr_curve[1]})
    pr_scores[clf] = pr_curve_df[pr_curve_df['recall'] >= 0.7]['precision'].max()

ans4 = repr(pandas.Series(pr_scores).sort_values(ascending=False).head(1).index[0])

print(ans1,"\n",ans2,"\n",ans3,"\n",ans4,"\n")

file_answer1 = open("accuracy_answer1.txt", "w")
file_answer2 = open("accuracy_answer2.txt", "w")
file_answer3 = open("accuracy_answer3.txt", "w")
file_answer4 = open("accuracy_answer4.txt", "w")
file_answer1.write(ans1)
file_answer2.write(ans2)
file_answer3.write(ans3)
file_answer4.write(ans4)
file_answer1.close()
file_answer2.close()
file_answer3.close()
file_answer4.close()

sys.exit(os.EX_OK) # code 0, all ok
