import pandas as pd
import numpy as np
from numpy import sort
from sklearn import metrics
import os
from copy import deepcopy
from matplotlib import pyplot as plt

from sklearn.cluster import AgglomerativeClustering, SpectralClustering
import manual_clustering


def get_hardcodings():
    cluster_overrides=[
                        ['com.belkatechnologies.clockmaker', 'com.g5e.hiddencity', 'com.g5e.romepg.ios', 'com.g5e.thehiddentreasurespg', 'com.gramgames.mergedragons', 'com.gramgames.mergemagic',
                         'com.bitmango.ap.jewelsmagicmysterymatch3', 'com.zynga.ozmatch', 'com.ivyios.jewelsclassic.prince', 'com.firstmagic.candycruise.free', 'com.gummy.candy.ios', 'com.superbox.ios.magiclamp',
                         'com.superbox.ios.jewelsjungle', 'com.bitmango.ap.lollipoplinkmatch', 'com.kudo.icecrush'],
                        ['com.foxnextgames.m3', 'com.mobilewaronline.wwr.submission', 'com.arenaofstarsonline.chof.submission', 'com.epicactiononline.ffxv.submission', 'com.nexters.titanhunters', 'com.kingsgroup.ss',
                         'com.ea.starwarscapital.bv', 'com.yottagames.mafiawar', 'zombie.survival.craft.z', 'com.beingame.zc.zombie.shelter.survival', 'com.camelgames.superking', 'com.more.dayzsurvival.ios',
                         'com.survival.dawn'],
                        ['com.belkatechnologies.fe', 'com.zynga.farmville2countryescape', 'com.zynga.farmvilletropicescape', 'com.gameinsight.thetribezhd', 'com.melsoftgames.familyisland', 'com.vizor-apps.klondike'],
                        ['app.kryds.game', 'com.randomlogicgames.crossword', 'com.etermax.guesscrack', 'com.fanatee.cody'],
                        ['com.matchington.mansion', 'com.crowdstar.covethome', 'com.storm8.hdpropertybrothers', 'com.fun-gi.house-flip', 'com.ivmob.interior', 'in.loop.hdmakeover', 'com.cookapps.ff.moderncity',
                         'com.smartfunapps.home.word', 'com.playrix.gardenscapes-sweethome', 'com.zentertain.homedesign', 'com.gamegos.mobile.manorcafe', 'dk.tactile.lilysgarden'],
                        ['com.redemptiongames.sugar', 'com.nordcurrent.canteenhd', 'net.ajoy.kf', 'com.glu.dashtown', 'com.ldw.cooking', 'com.bigfishgames.cookingempireuniversalf2p', 'com.melesta.mycafe',
                         'com.dragonplus.cookingfrenzy', 'com.playfirst.cookingdashx', 'com.nukeboxstudios.foodtruckchefcookinggame', 'com.digitalkingdom.cookinghome', 'com.flybirdgames.cookinghot',
                         'com.casualjoygames.cooking.kitchen', 'com.newvoy.cookingvoyage'],
                    ]

    hardcoded_clusters = set()
    for ls in cluster_overrides:
        hardcoded_clusters = hardcoded_clusters.union(set(ls))

    return hardcoded_clusters


# def get_ordered_paths(dir_path = "Downloads/cluster_data/clusters"):

#     paths = os.listdir(dir_path)
#     try:
#         paths.remove('.DS_Store')
#     except:
#         pass

#     paths = [os.path.join(dir_path, p) for p in paths]

#     path_dict = {path[-8:-4]:path for path in paths}
#     path_dict

#     ordered_paths = []
#     for key in sort(list(path_dict.keys())):
#         ordered_paths.append(path_dict[key])


#     dates = [o[-8:-4] for o in ordered_paths]

#     return ordered_paths, dates



# helpers for get_metrics()

def get_clusters(data): # get {app:cluster idx} from df
    clusters = {}
    for i in range(data.shape[0]):
        try:
            cluster = data.loc[i]
            apps_in_cluster = []
            for app in cluster:
                if type(app) == str:
                    clusters[app] = i
        except:
            pass
#             print("row {} not in data".format(i))
    return clusters

# get df from path
def get_data(path="Downloads/ios_install_clusters_c541_ios_4b4af316-bc7b-11ea-983f-42010a80005f.csv", verbose=False):
    data = pd.read_csv(path, index_col='0')
    if verbose:
        print(data['1'])
    del data['1']
    return data

# get dfs from paths
def get_all_data(paths):
    data = []
    for path in paths:
        d = get_data(path)
        data.append(d)
    return data

# get intersection of clusters' apps, subtract hardcoded apps
def get_apps_set(cluster1, cluster2, get_app_discrepancies, app_discrepancies=[], apps_added=[], apps_dropped=[]): # takes clusters as {app name:cluster index} dict

    hardcoded_clusters = get_hardcodings()
    # compile set of all relevant apps
    cluster1_apps = set(cluster1.keys()) - hardcoded_clusters
    cluster2_apps = set(cluster2.keys()) - hardcoded_clusters
    apps_set = list(cluster1_apps&cluster2_apps)
    # construct dropped/added apps
    if get_app_discrepancies:
        dropped_apps = len(cluster1_apps - cluster2_apps)
        new_apps = len(cluster2_apps - cluster1_apps)
        apps_added.append(new_apps)
        apps_dropped.append(dropped_apps)
        apps_discrepancy = len(cluster1_apps|cluster2_apps) - len(apps_set)
        assert apps_discrepancy == new_apps + dropped_apps, (apps_discrepancy, new_apps + dropped_apps)
        app_discrepancies.append(apps_discrepancy)
    # return apps_set
    return apps_set, app_discrepancies, apps_added, apps_dropped

# run metrics for clustering pair
def get_metrics_for(clustering_a, clustering_b):
    clustering_metrics = []
#     print("ca: {}\ncb: {}".format(clustering_a, clustering_b))
    ami = metrics.adjusted_mutual_info_score(clustering_a, clustering_b)
    ari = metrics.adjusted_rand_score(clustering_a, clustering_b)
    v_score = metrics.v_measure_score(clustering_a, clustering_b)
    clustering_metrics.append(ami)
    clustering_metrics.append(ari)
    clustering_metrics.append(v_score)
    return clustering_metrics

def mess_up(cluster_dict, p):
    apps = list(cluster_dict.keys())
    keys_to_replace = np.random.choice(a=apps, size=int(p*len(apps)), replace=False)
    max_cluster = max(cluster_dict.values())
    for key in keys_to_replace:
        cluster_dict[key] = np.random.randint(0,int(max_cluster))
    return cluster_dict




# get metrics for all adjacent pairs in dir_path
def get_metrics(dir_path='Downloads/cluster_data/features/', p=.20, get_app_discrepancies=True, first_clustering=AgglomerativeClustering, second_clustering=SpectralClustering, limit_to_n=None):
    cluster_data, _, dates = manual_clustering.get_clusterings_info(limit_to_n=limit_to_n, clustering_type=first_clustering)

    if second_clustering:
        second_cluster_data, _, dates2 = manual_clustering.get_clusterings_info(limit_to_n=limit_to_n, clustering_type=second_clustering)
        assert all(dates2 == dates)
        cluster_data2 = []
        for i in range(min(len(second_cluster_data), len(cluster_data))):
            cluster_data2.append(cluster_data[i])
            cluster_data2.append(second_cluster_data[i])
        cluster_data = cluster_data2
    
    metrics_ls = []
    app_discrepancies, apps_added, apps_dropped = [], [], []
    for i in range(len(cluster_data) - 1):
        j = i+1
        cluster0 = cluster_data[i]
        cluster1 = cluster_data[j]
        apps, app_discrepancies, apps_added, apps_dropped = get_apps_set(cluster0, cluster1, get_app_discrepancies, app_discrepancies, apps_added, apps_dropped)
        cluster0_ls, cluster1_ls = [], []
        for app in apps:
            cluster0_ls.append(cluster0[app])
            cluster1_ls.append(cluster1[app])
        metric = get_metrics_for(cluster0_ls, cluster1_ls)
        metrics_ls.append(metric)
    return metrics_ls, app_discrepancies, apps_added, apps_dropped, dates

# helper for show_metrics

def plot_data(data, title="AMIs", plt_axis=[None,None,None,None], x_labels=None, plot_with='o'):
    if type(x_labels) == type(None):
        x_labels = dates[1:]
    fig, ax = plt.subplots()
    ax.set_xticklabels(x_labels)
    ax.set_xticks(list(range(0,len(x_labels)+1)))

    plt.plot(data, plot_with)
    plt.axis(plt_axis)
    plt.suptitle(title)

def show_metrics(app_discrepancies, apps_added, apps_dropped, metrics_ls, dates):
    all_amis, all_aris, all_v_scores = [], [], []
    for metric_ls in metrics_ls:
        amis, aris, v_scores = [], [], []
        for ami, ari, v in metric_ls:
            amis.append(ami)
            aris.append(ari)
            v_scores.append(v)
        all_amis.append(amis)
        all_aris.append(aris)
        all_v_scores.append(v_scores)
#     print("amis: {}\naris: {}\n v: {}".format(all_amis, all_aris, all_v_scores))
    plot_data(data=np.transpose(all_amis), title="AMIs", plt_axis=[None,None,0,1], x_labels=dates[1:], plot_with='o')
    plot_data(data=np.transpose(all_aris), title="ARIs", plt_axis=[None,None,0, 1], x_labels=dates[1:])
    plot_data(data=np.transpose(all_aris), title="ARIs", plt_axis=[None,None,-1, 1], x_labels=dates[1:])
    plot_data(data=np.transpose(all_v_scores), title="V-Scores", plt_axis=[None,None,0, 1], x_labels=dates[1:])

    plot_data(data=app_discrepancies, title="# Apps Different", x_labels=dates[1:])
    plot_data(data=apps_added, title="# Apps Added", x_labels=dates[1:])
    plot_data(data=app_discrepancies, title="# Apps Dropped", x_labels=dates[1:])


def run_script(first_clustering=AgglomerativeClustering, second_clustering=SpectralClustering, limit_to_n=None, p=0):

    metrics_ls, app_discrepancies, apps_added, apps_dropped, dates = get_metrics(p=p, first_clustering=first_clustering, second_clustering=second_clustering, limit_to_n=limit_to_n)
    return app_discrepancies, apps_added, apps_dropped, metrics_ls, dates

def main(limit_to_n=None):
    all_metrics_ls = []
    print('ami, ari, v_score')
    # for p in [0]:#np.array(range(0,6))/5.:
    _,_,_, metrics_ls, dates = run_script(first_clustering=AgglomerativeClustering, second_clustering=SpectralClustering, limit_to_n=limit_to_n)
    all_metrics_ls.append(metrics_ls[1::2])
    app_discrepancies, apps_added, apps_dropped, metrics_ls, dates = run_script(first_clustering=AgglomerativeClustering,  second_clustering=None, limit_to_n=limit_to_n)
    all_metrics_ls.append(metrics_ls)
    show_metrics(app_discrepancies, apps_added, apps_dropped, all_metrics_ls, dates)
    return app_discrepancies, apps_added, apps_dropped, all_metrics_ls, dates




