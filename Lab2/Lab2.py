import sys
import numpy as np
import scipy.stats as ss
import pandas as pd
import csv
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from sklearn import metrics

import random
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap,MDS
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import cluster as Kcluster, metrics as SK_Metrics




#Function for random sampling of data
def random_sampling(data_frame, fraction):
    rows = random.sample(data_frame.index, (int)(len(data_frame)*fraction))
    print len(rows)
    return data_frame.ix[rows]

#Function for finding the optimum K using elbow curve
def new_elbow(inputData):
    K = range(1,10)
    KM = [KMeans(n_clusters=k).fit(inputData) for k in K]
    centroids = [k.cluster_centers_ for k in KM]
    D_k = [cdist(inputData, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]
    avgWithinSS = [sum(d)/inputData.shape[0] for d in dist]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(inputData)**2)/inputData.shape[0]
    bss = tss-wcss
    kIdx = 3-1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
    markeredgewidth=2, markeredgecolor='g', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Average Mean Square error')
    plt.title('Elbow curve for KMeans clustering')
    plt.show()

#Function for adaptive sampling of data using optimum K value
def adaptive_sampling(data_frame, cluster_count,fraction):
    k_means = Kcluster.KMeans(n_clusters=cluster_count)
    k_means.fit(data_frame)
    data_frame['label'] = k_means.labels_
    adaptiveSampleRows = []

    for i in range(cluster_count):
         adaptiveSampleRows.append(data_frame.ix[random.sample(data_frame[data_frame['label'] == i].index, (int)(len(data_frame[data_frame['label'] == i])*fraction))])


    adaptiveSample = pd.concat(adaptiveSampleRows)
    del adaptiveSample['label']
    print len(adaptiveSample)

    return adaptiveSample

# Function to perform MDS on reduced data
def find_MDS(dataframe, type):
    dis_mat = SK_Metrics.pairwise_distances(dataframe, metric = type)
    mds = MDS(n_components=2, dissimilarity='precomputed')
    return pd.DataFrame(mds.fit_transform(dis_mat))

#Function to generate necessary csv files
def create_files(random_sample, stratified_sample, file_name):
    data_directory = 'data/'

    random_sample['type'] = pd.Series('1', index=random_sample.index)
    stratified_sample['type'] = pd.Series('2', index=stratified_sample.index)
    if len(random_sample.columns) == 3:
        random_sample.columns = ['x','y','type']
        stratified_sample.columns = ['x','y','type']

    sample = pd.concat([random_sample, stratified_sample])

    file_name = data_directory + file_name
    sample.to_csv(file_name, sep=',', index=False)



def calculate_values(random_sample, adaptive_sample,function,file_name):
    create_files(function(random_sample), function(adaptive_sample),file_name +".csv")

def write_csv(random_sample, adaptive_sample):

    random_sample.to_csv("data/random_sample.csv")
    adaptive_sample.to_csv("data/adaptive_sample.csv")




def main():
    cols = pd.read_csv("cwurData.csv", nrows=1).columns
    benefitDF = pd.read_csv("cwurData.csv", usecols=[3,4,5,6,7,8,9,11,12])
    print benefitDF.head(10)
    benefitDF = benefitDF.fillna(0)
    new_elbow(benefitDF)
    optimalK = 3
    random_sample = random_sampling(benefitDF, 0.2)
    adaptive_sample = adaptive_sampling(benefitDF,optimalK, 0.2)
    write_csv(random_sample, adaptive_sample)

    mds_types = ["euclidean", "correlation"]
    for type_mds in mds_types:
        create_files(find_MDS(random_sample, type_mds), find_MDS(adaptive_sample, type_mds), type_mds + ".csv")


main()
