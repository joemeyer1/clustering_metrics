from sklearn.cluster import AgglomerativeClustering, SpectralClustering
import os
import pandas as pd
import numpy as np
from numpy import sort


# get clustering_type clusterings as {app:cluster_i} and list of lists of cluster_i's
def get_clusterings_info(similarity_matrices_path = 'Downloads/cluster_data/features/', limit_to_n=None, clustering_type=AgglomerativeClustering, swap_p=0):
    all_clusters, apps_arrays, dates = get_clusterings(similarity_matrices_path, limit_to_n, clustering_type=clustering_type, swap_p=swap_p)
    app_cluster_dicts = get_app_cluster_dict(all_clusters, apps_arrays)
    return app_cluster_dicts, all_clusters, dates


# HELPERS FOR get_clusterings_info()

def get_clusterings(similarity_matrices_path = 'Downloads/cluster_data/features/', limit_to_n=None, clustering_type=AgglomerativeClustering, swap_p=0):
    matrix_arrays, apps_arrays, dates = dir_to_dfs(similarity_matrices_path)[:,:limit_to_n]
    all_clusters = get_all_clusters(matrix_arrays, clustering_type=clustering_type, swap_p=swap_p)
    apps_clusters_dict = {}
    assert len(matrix_arrays) == len(apps_arrays)
        
    return all_clusters, apps_arrays, dates


def get_app_cluster_dict(all_clusters, all_apps):
    assert len(all_clusters) == len(all_apps)
    app_cluster_dicts = []
    for i in range(len(all_apps)):
        apps = all_apps[i]
        clusters = all_clusters[i]
        assert len(clusters) == len(apps)
        app_cluster_dict = {}
        for j in range(len(apps)):
            app = apps[j]
            cluster = clusters[j]
            app_cluster_dict[app] = cluster
        app_cluster_dicts.append(app_cluster_dict)
    return app_cluster_dicts


# HELPERS FOR get_clusterings()


def dir_to_dfs(similarity_matrices_path = 'Downloads/cluster_data/features/'):
    matrix_arrays = []
    apps_arrays = []
    if type(similarity_matrices_path) == str:
        matrix_paths, dates = dir_to_matrix_paths(similarity_matrices_path)
    else:
        matrix_paths = similarity_matrices_path
    for matrix_path in matrix_paths:
        matrix_df = path_to_df(matrix_path)
        apps_arrays.append(matrix_df.columns)
        matrix_array = df_to_array(matrix_df)
        matrix_arrays.append(matrix_array)
        print('.', end='')
    print("got arrays")
    return np.array([matrix_arrays, apps_arrays, dates])


def get_all_clusters(matrix_arrays, clustering_type=AgglomerativeClustering, swap_p=0):
    all_clusters = []
    print("messing up w probability {}".format(swap_p))
    for matrix_array in matrix_arrays:
        clusters = get_clusters(matrix_array, clustering_type=clustering_type, swap_p=swap_p)
        all_clusters.append(clusters)
        print('.', end='')
    print("got clusterings from arrays")
    return all_clusters


# HELPERS FOR dir_to_dfs()


def dir_to_matrix_paths(dir_path = "Downloads/cluster_data/features/"):

    paths = os.listdir(dir_path)
    try:
        paths.remove('.DS_Store')
    except:
        pass

    paths = [os.path.join(dir_path, p) for p in paths]

    path_dict = {path[-8:-4]:path for path in paths}
    path_dict

    ordered_paths = []
    for key in sort(list(path_dict.keys())):
        ordered_paths.append(path_dict[key])
    
    dates = [o[-8:-4] for o in ordered_paths]

    return ordered_paths, dates


def path_to_df(csv_path):
    similarity_matrix_df = pd.read_csv(csv_path, index_col='ID')
    del similarity_matrix_df['Package Name']
    return similarity_matrix_df

def df_to_array(matrix_df):
    return matrix_df.to_numpy()


# HELPERS FOR get_all_clusters()


def get_clusters(matrix_array: np.ndarray, clustering_type=AgglomerativeClustering, n_clusters=20, n_components=700, clustering_delta=1.0, swap_p=0):

    # Note: it is important to use float64 for the algorithms downstream
    matrix_array = mess_up_correlations(matrix_array, swap_p=swap_p)
    normalized_correlation_matrix = matrix_array.astype(np.float64)
    normalized_correlation_matrix = np.around(normalized_correlation_matrix, 4)

    normalized_correlation_matrix[normalized_correlation_matrix > 1] = 1
    distance_matrix = np.sqrt(2. * (1 - normalized_correlation_matrix))


    # Clustering
    if clustering_type == SpectralClustering:
        similarity_matrix = np.exp(- distance_matrix ** 2 / (2. * clustering_delta ** 2))
        clusters = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                assign_labels='discretize',
                n_components=n_components,
                random_state=32,
        ).fit_predict(similarity_matrix)
    else:
        dist_quantile = np.quantile(distance_matrix, q=np.arange(0, 1, 0.01))
        distance_threshold_dist_quantile = 8
        clusters = AgglomerativeClustering(
            n_clusters=None,
            affinity='precomputed',
            linkage='complete',
            distance_threshold=dist_quantile[distance_threshold_dist_quantile]
        ).fit_predict(distance_matrix)

    return clusters


# HELPER FOR get_clusters()

def mess_up_correlations(correlation_matrix: np.ndarray, swap_p=.2):
    # raise Exception("corr mat: ", correlation_matrix)
    xcoords = list(range(len(correlation_matrix)))
    ycoords = list(range(len(correlation_matrix[0])))
    coords = []
    for x in xcoords:
        for y in ycoords:
            coords.append((x,y))
    xys = np.random.choice(a=coords, size=int(swap_p*len(coords)), replace=False)
    for i in range(0, len(xys)-1,2):
        j = i+1
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        correlation_matrix[x1][y1], correlation_matrix[x2][y2] = correlation_matrix[x2][y2], correlation_matrix[x1][y1]
    return correlation_matrix





