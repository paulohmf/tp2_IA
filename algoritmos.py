import numpy as np
import pandas as pd
import random
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

#random.seed(42)
#random.seed(4)
random.seed(5) #fav

"""## Algoritmos

### KNN
"""

def euclidean_distance_knn(x_row, y_row, features):
  x = x_row[features].values
  y = y_row[features].values
  sum = 0
  for xi,yi, in zip(x,y):
    sum += np.power(xi-yi, 2)
  return np.sqrt(sum)

def KNN_predict(target, df_train, features, k=2):
  df_aux = df_train.copy()

  # Calculate trains euclidean distance
  df_aux["target_distance"] = df_aux.apply(lambda x: euclidean_distance_knn(x, target, features), axis=1)
  # Sort values by target_distance
  df_aux = df_aux.sort_values(by="target_distance")
  # Get KNN
  df_aux = df_aux.iloc[:k]
  # Group by label and get most frequente label
  predict = df_aux.groupby("Label").count().iloc[0].name

  return predict

"""
### k-means
"""

def euclidean_distance_kmeans(x, y_row, features):
  y = y_row[features].values
  sum = 0
  for xi,yi, in zip(x,y):
    sum += np.power(xi-yi, 2)
  return np.sqrt(sum)

def find_closest_centroid(x_row, dist_labels):
    row = x_row[dist_labels].values
    pos_min = np.where(row == np.amin(row))[0]
    return pos_min[0]

def instanciate_random_centroids(df_aux, features, k):
    centroids = []
    for ki in range(k):
      centroid = []
      for i, feature in zip(range(len(features)), features):
        f_min, f_max = np.min(df_aux[feature]), np.max(df_aux[feature])
        xi = random.uniform(f_min, f_max)
        centroid.append(xi)
      centroids.append(np.array(centroid))
    return centroids
    
def kmeans_train(df_train, features, k=3, max_epoch=100):
  df_aux = df_train.copy()

  centroids = instanciate_random_centroids(df_aux, features, k)

  df_aux["centroid"] = [-1 for i in range(len(df_aux))]
  stop = False
  epoch = -1
  while(not stop):
    epoch += 1
    ##print("Epoch:", epoch)
    df_aux["old_centroid"] = df_aux["centroid"].copy()

    # Calculate distance from row to each centroids
    for ki in range(k):
      col_name = "dist_" + str(ki)
      df_aux[col_name] = df_aux.apply(lambda x: euclidean_distance_kmeans(centroids[ki], x, features), axis=1)

    # Determine each row's closest centroids
    dist_labels = ["dist_" + str(ki) for ki in range(k)]
    df_aux["centroid"] = df_aux.apply(lambda x: find_closest_centroid(x, dist_labels), axis=1)

    # Update centroids
    new_centroids = []
    for ki in range(k):
      new_centroids.append(df_aux[df_aux["centroid"] == ki].mean()[features].values)
    
    # If no label has changed or epochs runned out stop
    if(np.sum(df_aux["old_centroid"] != df_aux["centroid"]) == 0):
      stop = True
    if(epoch > max_epoch):
      stop = True

  return centroids

def kmeans_predict(row, centroids, features):
  k = len(centroids)

  distances = []
  for ki in range(k):
      distances.append(euclidean_distance_kmeans(centroids[ki], row, features))
  distances = np.array(distances)

  pos_min = np.where(distances == np.amin(distances))[0][0]
  return pos_min
