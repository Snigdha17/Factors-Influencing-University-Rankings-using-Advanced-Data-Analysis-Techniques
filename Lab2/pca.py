import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler


#Function to estimate top three loadings
def find_top_three(data_frame, component_count):
    loadings, sample = get_squared_loadings(data_frame, component_count)
    loadings_sorted = loadings.sort_values(by='pca_loadings', ascending=False)

    return pd.DataFrame({
        loadings_sorted['index_name'].iloc[0]: data_frame[loadings_sorted['index_name'].iloc[0]],
        loadings_sorted['index_name'].iloc[1]: data_frame[loadings_sorted['index_name'].iloc[1]],
        loadings_sorted['index_name'].iloc[2]: data_frame[loadings_sorted['index_name'].iloc[2]]
    })


#Function to generate Scree Plot
def createScreePlot(inputSamples):
    stdinput = StandardScaler().fit_transform(inputSamples)
    cov_mat = np.cov(stdinput.T)
    eig_vals1, eig_vecs1 = np.linalg.eig(cov_mat)
    cor_mat1 = np.corrcoef(stdinput.T)
    eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

    y = eig_vals
    x = np.arange(len(y)) + 1

    df_eig = pd.DataFrame(eig_vals)
    df_eig.columns = ["eigan_values"]
    df_pca = pd.DataFrame(x)
    df_pca.columns = ["PCA_components"]
    sample = df_eig.join([df_pca])

    plt.figure()
    plt.plot(x, y, "o-")
    plt.plot(x[1], y[1], marker='o', markersize=12,
    markeredgewidth=2, markeredgecolor='g', markerfacecolor='None')
    plt.xticks(x, ["PC" + str(i) for i in x], rotation=60)
    plt.ylabel("Eigan Value")
    plt.show()

#Function to generate Squared Loadings of data
def get_squared_loadings(dataframe, intrinsic):
    std_input = StandardScaler().fit_transform(dataframe)
    pca = PCA(n_components=intrinsic)
    pca.fit_transform(std_input)

    loadings = pd.DataFrame(pca.components_)

    squared_loadings = []
    a = np.array(loadings)
    a = a.transpose()
    for i in range(len(a)):
        squared_loadings.append(np.sum(np.square(a[i])))
    df_attributes = pd.DataFrame(pd.DataFrame(dataframe).columns)
    df_attributes.columns = ["attributes"]
    df_sqL = pd.DataFrame(squared_loadings)
    df_sqL.columns = ["squared_loadings"]
    sample = df_attributes.join([df_sqL])
    sample = sample.sort_values(["squared_loadings"], ascending=[False])

    sample.plot(kind='bar', x = "attributes", y = "squared_loadings", figsize = (8,7), color = "green")
    plt.ylabel('Squared Loadings')
    plt.xlabel('Attributes')
    plt.title('Picking the top three attributes')
    plt.tight_layout()
    plt.show()
    sample.to_csv("data/squared_loadings.csv", sep=',')

    loadings_mat = loadings.applymap(np.square)
    loadings_final = loadings_mat.transpose().sum(axis=1)
    return pd.DataFrame({
        'index_name': dataframe.columns,
        'pca_loadings': loadings_final
    }), sample


# Functions to create csv files
def calculate_values(random_sample, stratified_sample,function,file_name, component_count):
    create_files(function(random_sample, component_count), function(stratified_sample, component_count), file_name + '.csv')


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

#Function to generate csv files for scatter matrix
def create_files_pcaloadings(random_sample, stratified_sample, file_name):
    data_directory = 'data/'
    filename = data_directory + file_name
    random_sample.to_csv(filename + '_random' + '.csv', sep = ',', index = False)
    stratified_sample.to_csv(filename + '_adaptive' + '.csv', sep = ',', index = False)


#Function to generate csv for data in top 2 PCA vectors
def analyse_pca(random_sample, stratified_sample):
    pca = PCA(n_components=2)
    pca_random = pd.DataFrame(pca.fit_transform(random_sample))
    pca_stratified = pd.DataFrame(pca.fit_transform(stratified_sample))
    create_files(pca_random, pca_stratified, "pca_output.csv")

def main():

    random = pd.read_csv("data/random_sample.csv")
    adaptive = pd.read_csv("data/adaptive_sample.csv")

    #createScreePlot(random)
    createScreePlot(adaptive)

    intrinsic = 2

    '''
    pca = PCA().fit(adaptive)
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()
    '''

    analyse_pca(random, adaptive) # For two components

    create_files_pcaloadings(find_top_three(random, intrinsic), find_top_three(adaptive, intrinsic), 'top_three_loadings')

main()