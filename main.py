# -*- coding: utf-8 -*-
"""tp2_IA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xubcU3vq6znpzVyO2po_zdYUJ9cL1GQ9

# Trabalho Prático 2 - Aprendizado de Máquina

**Paulo Henrique Maciel Fraga 2018054451**
"""
import pandas as pd
import json

from algoritmos import KNN_predict, kmeans_train, kmeans_predict

"""## Load de dados"""

columns_names = ["Sepal_lenght", "Sepal_width", "Petal_lenght","Petal_width","Label"]

df_aleatorio = pd.read_csv('data/iris_aleatório.csv', sep=';', header=None)
df_aleatorio.columns = columns_names

df_treino = pd.read_csv('data/iris_treino.csv', sep=';', header=None)
df_treino.columns = columns_names

df_teste = pd.read_csv('data/iris_teste.csv', sep=';', header=None)
df_teste.columns = columns_names


"""## Experimental"""

features = ["Sepal_lenght", "Sepal_width", "Petal_lenght","Petal_width"]

k = [2,3,8,32]
for ki in k:
  label = "KNN_guess_k" + str(ki)
  df_teste[label] = df_teste.apply(lambda x: KNN_predict(x, df_treino, features, k=ki), axis=1)

k = [2,3]
dict_result_kmeans = {}
for ki in k:
  centroids = kmeans_train(df_treino, features, k=ki)
  dict_result_kmeans[ki] = [[ci for ci in c] for c in centroids]
  label = "Kmeans_group_k" + str(ki)
  df_teste[label] = df_teste.apply(lambda x: kmeans_predict(x, centroids, features), axis=1)
  #print("------ k=", ki)
  #print("centroids :", centroids)

with open('outputs/centroids.txt', 'w') as f:
     f.write(json.dumps(dict_result_kmeans))

df_result = df_teste.drop(columns=features)
df_result.to_csv("outputs/resultados.csv")

