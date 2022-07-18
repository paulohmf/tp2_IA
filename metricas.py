import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib as mpl
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

## Load dos resultados

with open("outputs/centroids.txt") as f:
    dict_result_kmeans = json.load(f)

df_result = pd.read_csv('outputs/resultados.csv')

"""## Métricas - KNN"""

labels = pd.unique(df_result["Label"])

def get_confusion_matriz(alg):
  confusion = {}
  for l1 in labels:
    l1_result = {}
    for l2 in labels:
      count = df_result[(df_result["Label"] == l1) & (df_result[alg] == l2)].shape[0]
      l1_result[l2] = count
    confusion[l1] = l1_result
  
  return pd.DataFrame(confusion)

def acuracia(cf_matriz):
  right_guesses = np.sum([cf_matriz[cf_matriz.index == l][l] for l in labels])
  total = np.sum([np.sum(cf_matriz[l]) for l in labels])
  return right_guesses/total

def precisao(cf_matriz, label):
  right_guess_label = cf_matriz[cf_matriz.index == label][label].values[0]
  total = np.sum(cf_matriz[label].values)
  if(total == 0):
    return "nan"
  return right_guess_label/total

def revocacao(cf_matriz, label):
  right_guess_label = cf_matriz[cf_matriz.index == label][label].values[0]
  total = np.sum(cf_matriz[cf_matriz.index == label].values)
  if(total == 0):
    return "nan"
  return right_guess_label/total

def F1(cf_matriz, label):
  r = revocacao(cf_matriz, label)
  p = precisao(cf_matriz, label)
  if(r == "nan" or p == "nan"):
    return "nan"
  return 2*r*p/(r+p)

def analise(column):
  cf_matriz = get_confusion_matriz(column)
  print("----- " +column)
  print(cf_matriz)
  location = "outputs/matriz_" + column + ".txt"
  cf_matriz.to_latex(location)
  print()

  print("Acurácia :", acuracia(cf_matriz))
  print()

  final_summary = {}
  for l in labels:
    summary = {}
    summary["Precisão"] = precisao(cf_matriz, l)
    summary["Revocação"] = revocacao(cf_matriz, l)
    summary["F1"] =  F1(cf_matriz, l)
    final_summary[l] = summary
  final_summary = pd.DataFrame(final_summary)
  print(final_summary)

  location = "outputs/summary_" + column + ".txt"
  final_summary.to_latex(location)


"""#### K=2"""
column = "KNN_guess_k2"
analise(column)

"""#### K=3"""

column = "KNN_guess_k3"
analise(column)

"""#### K=8"""

column = "KNN_guess_k8"
analise(column)

"""#### K=32"""

column = "KNN_guess_k32"
analise(column)

"""## Gráficos k-means

### k=2
"""
print("----- k-means k=2")
print("Centroides:")
for i in dict_result_kmeans['2']:
  print(i)

groupby = df_result.groupby(by=["Label", "Kmeans_group_k2"]).count()[["KNN_guess_k2"]].rename(columns={'KNN_guess_k2':'Count'})
print(groupby)
groupby.to_latex("outputs/summary_kmeans_k2.txt")

pd.crosstab(df_result['Label'],df_result['Kmeans_group_k2']).plot.bar()
plt.title("Distribuição labels originais e grupos k=2")
plt.legend(bbox_to_anchor=(1.1, 0.9))
plt.xticks(rotation = 360)
plt.grid()
plt.savefig("outputs/k_means_k2.png")
print()

"""### k=3"""
print("----- k-means k=3")

print("Centroides:")
for i in dict_result_kmeans['3']:
  print(i)

groupby = df_result.groupby(by=["Label", "Kmeans_group_k3"]).count()[["KNN_guess_k2"]].rename(columns={'KNN_guess_k2':'Count'})
print(groupby)
groupby.to_latex("outputs/summary_kmeans_k3.txt")

pd.crosstab(df_result['Label'],df_result['Kmeans_group_k3']).plot.bar()
plt.title("Distribuição labels originais e grupos k=3")
plt.legend(bbox_to_anchor=(1.1, 0.9))
plt.xticks(rotation = 360)
plt.grid()
plt.savefig("outputs/k_means_k3.png")
