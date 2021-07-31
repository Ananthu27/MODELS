from random import uniform
from dataset import  forKMeans
import sys
import numpy as np

from general import eulidianDistance
FLOAT_MAX = sys.float_info.max

### K MEANS IMPLEMENTATION
def kMeans(k,df,columns,EPOC=10):

    min_max = {}
    for column in columns:
        min_max[column] = (df[column].min(),df[column].max())

    cluster_centers = []
    for disacrd in range(k):
        center = []
        for column in columns:
            center.append(uniform(min_max[column][0],min_max[column][1]))
        cluster_centers.append(center)

    for i in range(EPOC):
        df = assignCluster(cluster_centers,df,columns)
        cluster_centers = calcClusterCenters(df,columns)

    return cluster_centers, df

### FUNCTION TO ASSIGN DATA POINTS TO CLUSTER CENTERS
def assignCluster(cluster_centers,df,columns):
    if len(cluster_centers[0]) == len(columns):
        df['cluster'] = [None for i in range(len(df[columns[0]]))]
        for index,row in df.iterrows():
            min_distance = FLOAT_MAX
            min_index = None
            for i,center in enumerate(cluster_centers):
                distance = eulidianDistance(center,[row[column] for column in columns])
                if distance < min_distance:
                    min_distance = distance
                    min_index = i
            df.at[index,'cluster'] = min_index
    return df

### FUNCTION TO RECALCULATE CLUSTER CENTER
def calcClusterCenters(df,columns):
    cluster_centers = []
    clusters = np.sort(df['cluster'].unique())
    k = len(clusters)
    for i in range(k):
        temp = df[df.cluster==clusters[i]].head()
        cluster_centers.append([temp.x.mean(),temp.y.mean()])
    return cluster_centers

# df = forKMeans()
# columns = df.columns
# cc,df = kMeans(3,df,columns,100)

# import matplotlib.pyplot as plt

# colors=["#DDDDFF", "#DDFFDD", "#FFDDDD"]

# for index,row in df.iterrows():
#     plt.scatter(x=row['x'],y=row['y'],color=colors[row['cluster']])

# import pandas as pd
# ccdf = pd.DataFrame(cc,columns=columns)

# plt.scatter(x=ccdf['x'],y=ccdf['y'],color=["#000000" for i in range(len(ccdf['x']))])
# plt.show()