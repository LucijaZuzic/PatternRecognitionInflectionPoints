import pandas as pd
from utilities import *
import argparse
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import kneed
import os
import matplotlib.pyplot as plt
from time import time

# Define the parser

parser = argparse.ArgumentParser(description = "generate a fingerprint for multiple random trajectories of a specified length with the specified window size")

# Defining the number of trajectories, the default value is 1167.
# If the number of trajectories is not an integer greater than 0, trajectories are not generated.
# Trajectories in .csv files are stored in a .csv file in the "trajectories" subdirectory.

parser.add_argument("--trajectories", action = "store", dest = "trajectories", default = 1167, help = "trajectories to generate")

# Defining the number of longitude and latitude points, the default value is 40.
# If the number of longitude and latitude points is not an integer greater than 0, the default value is applied.
# Trajectory generation execution times are stored in a .csv file in the "results" subdirectory.

parser.add_argument("--points", action = "store", dest = "points", default = 40, help = "trajectory length")

# Defining the window size, the default value is 20.
# If the window size is not an integer greater than 0, the default value is applied.
# If the window size is larger than the number of longitude and latitude points,
# the window size is decreased to match the mumber of longitude and latitude points.
# Trajectory features with the specified window size are stored in a .csv file in the "results" subdirectory.
# Trajectory fingerprinting execution times with the specified window size are stored in a .csv file in the "results" subdirectory.

parser.add_argument("--window", action = "store", dest = "window", default = 20, help = "window size")

# Defining the maximum number of features for clustering, the default value is 200.
# Clustering outcomes for the specified window size and algorithm are stored in a .csv file in the "results" subdirectory.
# Clustering outcomes for the specified window size and algorithm are visualised using PCA and TSNE, with .pdf, .png and .svg plots stored in the "plots" subdirectory.
# Trajectory features with PCA and TSNE for the specified window size and algorithm are stored in a .csv file in the "results" subdirectory.

parser.add_argument("--features", action = "store", dest = "features", default = 200, help = "number of clustering features")

# Defining the minimum cluster size for DBSCAN clusters, the default value is 346.
# If the value is not an integer integer greater than 0 and less than or equal to the number of trajectories, DBSCAN clustering is not applied.

parser.add_argument("--neighbors", action = "store", dest = "neighbors", default = 346, help = "minimum DBSCAN cluster size")

# Defining the number of clusters for KMeans clustering, the default value is 2.
# If the value is not an integer greater than 1, KMeans clustering is not applied.

parser.add_argument("--clusters", action = "store", dest = "clusters", default = 2, help = "number of K-means clusters")

# The user can decide if they want to generate a plot showing the inflection points for each trajectory.
# The default value is false, so the plots will not be generated unless the parameter is specified.
# Plots showing the inflection points in .pdf, .png and .svg formats are stored in the "trajecotry_plots" subdirectory.

parser.add_argument("--plots", action = "store_true", dest = "plots", help = "generate inflection point plots")

# The user can decide if they want to read the output of trajectory fingerprinting in the command line.
# The default value is false, so only the clustering results will be printed in the command line.

parser.add_argument("--verbose", action = "store_true", dest = "verbose", help = "fingerprint command line output")

args = parser.parse_args()

plots = args.plots

verbose = args.verbose

try:
    trajectories = int(args.trajectories)
except:
    print("The number of trajectories (" + str(args.trajectories) + ") is not an integer greater than 0, trajectorires are not generated.")
    trajectories = 1167

if trajectories < 1:
    print("The number of trajectories (" + str(args.trajectories) + ") is not an integer greater than 0, trajectorires are not generated.")
    trajectories = 1167

try:
    points = int(args.points)
except:
    print("The number of points (" + str(args.points) + ") is not an integer greater than 0, using a default value of 40.")
    points = 40

if points < 1:
    print("The number of points (" + str(args.points) + ") is not an integer greater than 0, using a default value of 40.")
    points = 40

try:
    window = int(args.window)
except:
    print("The window size (" + str(args.window) + ") is not an integer greater than 0, using a default value of 20.")
    window = 20
    
if window < 1:
    print("The window size (" + str(args.window) + ") is not an integer greater than 0, using a default value of 20.")
    window = 20

if window > points:
    print("The window size (" + str(args.window) + ") is larger than the number of longitude and latitude points (" + str(args.points) + "), decreasing the window size to match the number of longitude and latitude points.")
    window = points
    
try:
    features = int(args.features)
except:
    print("The maximum number of features for clustering (" + str(args.features) + ") is not an integer greater than 0, using a default value of 200.")
    features = 200

if features < 1:
    print("The maximum number of features for clustering (" + str(args.features) + ") is not an integer greater than 0, using a default value of 200.")
    features = 200

time_dict_generating = {"task": [], "time": []}

if trajectories > 0:

    for trajectory in range(trajectories):

        # Generating a trajectory
        if verbose:
            print("Generating a trajectory:", trajectory + 1)

        starting_values =  {"longitude": 0, "latitude": 0}

        dictionary_write_previous = {"fields_" + varname: [] for varname in starting_values}

        probability = dict()
        probability_next = dict()
        probability_next_next = dict()

        time_now = time()
        for varname in starting_values:
            probability[varname] = load_object("probability/" + varname + "/probability_of_" + varname)
            probability_next[varname] = load_object("probability/" + varname + "/probability_of_" + varname + "_in_next_step")
            probability_next_next[varname] = load_object("probability/" + varname + "/probability_of_" + varname + "_in_next_next_step")
            dictionary_write_previous["fields_" + varname] = predict_probability(points, probability[varname], probability_next[varname], probability_next_next[varname])
        time_dict_generating["time"].append(time() - time_now)
        time_dict_generating["task"].append("generating_" + str(trajectory + 1))

        if verbose:
            print("Trajectory generation time:", time_dict_generating["time"][-1])

        for varname in starting_values:
            dictionary_write_previous["fields_" + varname] = [starting_values[varname] + sum(dictionary_write_previous["fields_" + varname][:ix]) for ix in range(len(dictionary_write_previous["fields_" + varname]))]
            data_frame_write = pd.DataFrame(dictionary_write_previous)
            if not os.path.isdir("trajectories"):
                os.makedirs("trajectories")
            data_frame_write.to_csv("trajectories/trajectory_" + str(trajectory + 1) + ".csv", index = False)

    time_dict_generating["time"].append(sum(time_dict_generating["time"]))
    time_dict_generating["task"].append("generating_all")
    time_dict_generating["time"].append(time_dict_generating["time"][-1] / trajectories)
    time_dict_generating["task"].append("generating_average")

    print("Total trajectory generation time:", time_dict_generating["time"][-2])
    print("Average trajectory generation time:", time_dict_generating["time"][-1])

    if not os.path.isdir("results"):
        os.makedirs("results")
    data_frame_time_generating = pd.DataFrame(time_dict_generating)
    data_frame_time_generating.to_csv("results/time_generating.csv", index = False)

trajectory_names = os.listdir("trajectories")
trajectory_numbers = sorted([int(trajectory_filename.split(".")[0].split("_")[1]) for trajectory_filename in trajectory_names])
trajectories = len(trajectory_names)

if trajectories > 1:

    time_dict_fingerprinting = {"task": [], "time": []}

    marker_dictionary = dict()
    marker_frequency = dict()
    marker_frequency_total = dict()

    for trajectory in trajectory_numbers:

        file_with_ride = pd.read_csv("trajectories/trajectory_" + str(trajectory) + ".csv")
        if verbose:
            print("Generating a trajectory fingerprint:", trajectory)
        longitudes = list(file_with_ride["fields_longitude"])
        latitudes = list(file_with_ride["fields_latitude"])

        # Generating a trajectory fingerprint

        if plots:
            prepare_figure()
            plt.title("Trajectory " + str(trajectory))
            plt.plot(longitudes, latitudes, label = "Trajectory", color = "b", zorder = 2)
        
        markers = []
        coordinates_of_inflection = []
        time_now = time()
        for x in range(0, len(longitudes) - window + 1):

            longitudes_segment = longitudes[x:x + window]
            latitudes_segment = latitudes[x:x + window]

            longitude_difference = [longitudes_segment[i] - longitudes_segment[0] for i in range(len(longitudes_segment))]
            latitude_difference = [latitudes_segment[i] - latitudes_segment[0] for i in range(len(latitudes_segment))] 
                
            angle_all = []
            radius_all = []
            for i in range(1, len(longitude_difference)):
                radius_all.append(np.sqrt(latitude_difference[i] ** 2 + longitude_difference[i] ** 2)) 
                angle_all.append(np.arctan2(latitude_difference[i], longitude_difference[i])) 

            longitude_new = [radius_all[i] * np.cos(angle_all[i] - angle_all[-1]) for i in range(len(radius_all))]
            latitude_new = [radius_all[i] * np.sin(angle_all[i] - angle_all[-1]) for i in range(len(radius_all))] 
            
            longitude_new.insert(0, 0)
            latitude_new.insert(0, 0)

            longitude_new, latitude_new = preprocess_longitude_latitude(longitude_new, latitude_new)

            longitude_sign = [longitude_new[i + 1] > longitude_new[i] for i in range(len(longitude_new) - 1)]
            latitude_sign = [latitude_new[i + 1] > latitude_new[i] for i in range(len(latitude_new) - 1)]

            longitude_change_sign = [longitude_sign[i + 1] != longitude_sign[i] for i in range(len(longitude_sign) - 1)]
            latitude_change_sign = [latitude_sign[i + 1] != latitude_sign[i] for i in range(len(latitude_sign) - 1)]

            marker = ""
            for i in range(len(longitude_change_sign)):
                if longitude_change_sign[i] or latitude_change_sign[i] :
                    computed_string_longitude = str(int(longitude_sign[i])) + str(int(longitude_sign[i + 1]))
                    computed_string_latitude = str(int(latitude_sign[i])) + str(int(latitude_sign[i + 1]))
                    integer_from_computed_string = int(computed_string_longitude + computed_string_latitude, base = 2)
                    marker += str(hex(integer_from_computed_string))[2:]
                    coordinates_of_inflection.append(x + i)
            markers.append(marker)
        
        time_dict_fingerprinting["time"].append(time() - time_now)
        time_dict_fingerprinting["task"].append("fingerprinting_" + str(trajectory + 1))

        if plots:
            coordinates_of_inflection = sorted(list(set(coordinates_of_inflection)))
            longitude_inflection = [longitudes[ix] for ix in coordinates_of_inflection]
            latitude_inflection = [latitudes[ix] for ix in coordinates_of_inflection]
            plt.scatter(longitude_inflection, latitude_inflection, label = "Inflection points", color = "r", zorder = 3)

            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.legend()
            if not os.path.isdir("trajectory_plots"):
                os.makedirs("trajectory_plots")
            plt.savefig("trajectory_plots/trajectory_plot_" + str(trajectory) + "_window_" + str(window) + ".pdf", bbox_inches = "tight")
            plt.savefig("trajectory_plots/trajectory_plot_" + str(trajectory) + "_window_" + str(window) + ".png", bbox_inches = "tight")
            plt.savefig("trajectory_plots/trajectory_plot_" + str(trajectory) + "_window_" + str(window) + ".svg", bbox_inches = "tight")
            plt.close()

        marker_frequency[trajectory] = {marker: markers.count(marker) / len(markers) for marker in set(markers)}
        
        if "number" not in marker_frequency_total:
            marker_frequency_total["number"] = []
        marker_frequency_total["number"].append(trajectory)

        for marker in marker_frequency_total:
            if marker != "number":
                marker_frequency_total[marker].append(0)

        for marker in marker_frequency[trajectory]:
            if verbose and marker_frequency[trajectory][marker] > 10 ** -2:
                print("Inflection point sequence code:", marker, "Frequency (%):", str(np.round(100 * marker_frequency[trajectory][marker], 2)) + "%")
            if marker not in marker_dictionary:
                marker_dictionary[marker] = 0
            marker_dictionary[marker] += marker_frequency[trajectory][marker]
            if marker not in marker_frequency_total:
                marker_frequency_total[marker] = [0 for ix in range(len(list(marker_frequency_total["number"])))]
            marker_frequency_total[marker][-1] = marker_frequency[trajectory][marker]

        if verbose:
            print("Trajectory fingerprinting time:", time_dict_fingerprinting["time"][-1])
    
    time_dict_fingerprinting["time"].append(sum(time_dict_fingerprinting["time"]))
    time_dict_fingerprinting["task"].append("fingerprinting_all")
    time_dict_fingerprinting["time"].append(time_dict_fingerprinting["time"][-1] / trajectories)
    time_dict_fingerprinting["task"].append("fingerprinting_average")

    print("Total trajectory fingerprinting time:", time_dict_fingerprinting["time"][-2])
    print("Average trajectory fingerprinting time:", time_dict_fingerprinting["time"][-1])

    if not os.path.isdir("results"):
        os.makedirs("results")
    data_frame_time_fingerprinting = pd.DataFrame(time_dict_fingerprinting)
    data_frame_time_fingerprinting.to_csv("results/time_fingerprinting_window_" + str(window) + ".csv", index = False)

    marker_dictionary_key_value = [(marker_dictionary[marker], marker) for marker in marker_dictionary]
    marker_dictionary_key_value = sorted(marker_dictionary_key_value, reverse = True)[:features]
    chosen_keys = sorted([key_value[1] for key_value in marker_dictionary_key_value])

    # Saving a file with trajectory features

    if not os.path.isdir("results"):
        os.makedirs("results")
    dictionary_chosen = {"number": marker_frequency_total["number"]}
    for marker in chosen_keys:
        dictionary_chosen[marker] = marker_frequency_total[marker]
    data_frame_chosen = pd.DataFrame(dictionary_chosen)
    data_frame_chosen.to_csv("results/trajectory_features_window_" + str(window) + ".csv", index = False)

    X = [] 
    for ix in range(len(marker_frequency_total["number"])):
        Xrow = []
        for marker in chosen_keys:
            Xrow.append(marker_frequency_total[marker][ix])
        X.append(Xrow)
    X = np.array(X)

    print("All trajectories")
    for frequency_marker in marker_dictionary_key_value:
        frequency, marker = frequency_marker
        if verbose and frequency / trajectories > 10 ** -2:
            print("Inflection point sequence code:", marker, "Total frequency (%):", str(np.round(100 * frequency / trajectories, 2)) + "%")

    print(np.shape(X)[0], "trajectories,", np.shape(X)[1], "features")

    # PCA for cluster illustrations

    X_components = PCA(n_components = 2, random_state = 42, svd_solver = "full").fit_transform(X)

    if not os.path.isdir("results"):
        os.makedirs("results")
    new_data_PCA = {"number": marker_frequency_total["number"],
                    "PCA_feature_1": [X_components[ix][0] for ix in range(len(marker_frequency_total["number"]))], 
                    "PCA_feature_2": [X_components[ix][1] for ix in range(len(marker_frequency_total["number"]))]}
    data_frame_PCA = pd.DataFrame(new_data_PCA)
    data_frame_PCA.to_csv("results/PCA_features_window_" + str(window) + ".csv", index = False)

    # TSNE for cluster illustrations

    X_embedded = TSNE(n_components = 2, perplexity = 30, random_state = 42, init = "pca").fit_transform(X)

    if not os.path.isdir("results"):
        os.makedirs("results")
    new_data_TSNE = {"number": marker_frequency_total["number"],
                    "TSNE_feature_1": [X_embedded[ix][0] for ix in range(len(marker_frequency_total["number"]))], 
                    "TSNE_feature_2": [X_embedded[ix][1] for ix in range(len(marker_frequency_total["number"]))]}
    data_frame_TSNE = pd.DataFrame(new_data_TSNE)
    data_frame_TSNE.to_csv("results/TSNE_features_window_" + str(window) + ".csv", index = False)

    neighbors = 0
    try:
        neighbors = int(args.neighbors)
    except:
        print("The minimum number of neighbors for DBSCAN clusters (" + str(args.neighbors) + ") is not an integer greater than 0 and less than or equal to the number of trajectories, DBSCAN clustering is not applied.")
        neighbors = 0

    if neighbors > np.shape(X)[0] or neighbors < 1:
        print("The minimum number of neighbors for DBSCAN clusters (" + str(args.neighbors) + ") is not an integer greater than 0 and less than or equal to the number of trajectories, DBSCAN clustering is not applied.")
        neighbors = 0

    if neighbors:  

        # k-nn to determine the DBSCAN epsilon parameter

        time_dict_clustering_DBSCAN = {"task": [], "time": []}  

        time_now = time()
        nearest_neighbors = NearestNeighbors(n_neighbors = neighbors)
        neighbors_fit = nearest_neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        time_dict_clustering_DBSCAN["time"].append(time() - time_now)
        time_dict_clustering_DBSCAN["task"].append("nearest_neighbors_DBSCAN_neighbors_" + str(neighbors) + "_window_" + str(window))

        print("k-nn time:", time_dict_clustering_DBSCAN["time"][-1])

        distances_sorted = np.sort(distances, axis = 0)
        distances_subset = distances_sorted[:, 1]

        # Finding the knee point

        time_now = time()
        kneedle = kneed.KneeLocator([ix for ix in range(len(distances_subset))], distances_subset, curve = "concave", direction = "decreasing")
        time_dict_clustering_DBSCAN["time"].append(time() - time_now)
        time_dict_clustering_DBSCAN["task"].append("knee_DBSCAN_neighbors_" + str(neighbors) + "_window_" + str(window))
        knee_point = kneedle.knee
        knee_point_dist = distances_subset[knee_point]
        
        print("Knee discovery time:", time_dict_clustering_DBSCAN["time"][-1])

        # Finding the elbow point

        time_now = time()
        kneedle = kneed.KneeLocator([ix for ix in range(len(distances_subset))], distances_subset, curve = "convex", direction = "increasing")
        time_dict_clustering_DBSCAN["time"].append(time() - time_now)
        time_dict_clustering_DBSCAN["task"].append("elbow_DBSCAN_neighbors_" + str(neighbors) + "_window_" + str(window))
        elbow_point = kneedle.elbow
        elbow_point_dist = distances_subset[elbow_point]
        
        print("Elbow discovery time:", time_dict_clustering_DBSCAN["time"][-1])

        # DBSCAN using the knee point

        if knee_point_dist:
            print("Knee:", knee_point, knee_point_dist)
            time_now = time()
            DBSCAN_knee = DBSCAN(min_samples = neighbors, eps = knee_point_dist).fit(X)
            time_dict_clustering_DBSCAN["time"].append(time() - time_now)
            time_dict_clustering_DBSCAN["task"].append("DBSCAN_knee_" + str(neighbors) + "_window_" + str(window))
            list_labels_DBSCAN_knee = list(DBSCAN_knee.labels_)

            print("DBSCAN clustering time (knee):", time_dict_clustering_DBSCAN["time"][-1])

            label_set_knee = set(list_labels_DBSCAN_knee)
            number_of_clusters_knee = len(label_set_knee)
            print("DBSCAN clusters (knee):", number_of_clusters_knee)
            for cluster_marker in sorted(list(label_set_knee)):
                print("Cluster marker:", cluster_marker, "Cluster size:", list_labels_DBSCAN_knee.count(cluster_marker))

            prepare_figure()
            plt.title("DBSCAN clusters (knee)\nneighbors: " + str(neighbors) + ", window: " + str(window))
            for cluster_marker in sorted(list(label_set_knee)):
                x_values_PCA = [new_data_PCA["PCA_feature_1"][ix] for ix in range(len(list_labels_DBSCAN_knee)) if list_labels_DBSCAN_knee[ix] == cluster_marker]
                y_values_PCA = [new_data_PCA["PCA_feature_2"][ix] for ix in range(len(list_labels_DBSCAN_knee)) if list_labels_DBSCAN_knee[ix] == cluster_marker]
                plt.scatter(x_values_PCA, y_values_PCA, label = str(cluster_marker + 1), zorder = cluster_marker + 3)
            plt.xlabel("PCA feature 1")
            plt.ylabel("PCA feature 2")
            plt.legend()
            if not os.path.isdir("plots/PCA/DBSCAN/knee"):
                os.makedirs("plots/PCA/DBSCAN/knee")
            plt.savefig("plots/PCA/DBSCAN/knee/PCA_DBSCAN_knee_neighbors_" + str(neighbors) + "_window_" + str(window) + ".pdf", bbox_inches = "tight")
            plt.savefig("plots/PCA/DBSCAN/knee/PCA_DBSCAN_knee_neighbors_" + str(neighbors) + "_window_" + str(window) + ".svg", bbox_inches = "tight")
            plt.savefig("plots/PCA/DBSCAN/knee/PCA_DBSCAN_knee_neighbors_" + str(neighbors) + "_window_" + str(window) + ".png", bbox_inches = "tight")
            plt.close()

            prepare_figure()
            plt.title("DBSCAN clusters (knee)\nneighbors: " + str(neighbors) + ", window: " + str(window))
            for cluster_marker in sorted(list(label_set_knee)):
                x_values_TSNE = [new_data_TSNE["TSNE_feature_1"][ix] for ix in range(len(list_labels_DBSCAN_knee)) if list_labels_DBSCAN_knee[ix] == cluster_marker]
                y_values_TSNE = [new_data_TSNE["TSNE_feature_2"][ix] for ix in range(len(list_labels_DBSCAN_knee)) if list_labels_DBSCAN_knee[ix] == cluster_marker]
                plt.scatter(x_values_TSNE, y_values_TSNE, label = str(cluster_marker + 1), zorder = cluster_marker + 3)
            plt.xlabel("TSNE feature 1")
            plt.ylabel("TSNE feature 2")
            plt.legend()
            if not os.path.isdir("plots/TSNE/DBSCAN/knee"):
                os.makedirs("plots/TSNE/DBSCAN/knee")
            plt.savefig("plots/TSNE/DBSCAN/knee/TSNE_DBSCAN_knee_neighbors_" + str(neighbors) + "_window_" + str(window) + ".pdf", bbox_inches = "tight")
            plt.savefig("plots/TSNE/DBSCAN/knee/TSNE_DBSCAN_knee_neighbors_" + str(neighbors) + "_window_" + str(window) + ".svg", bbox_inches = "tight")
            plt.savefig("plots/TSNE/DBSCAN/knee/TSNE_DBSCAN_knee_neighbors_" + str(neighbors) + "_window_" + str(window) + ".png", bbox_inches = "tight")
            plt.close()

            if not os.path.isdir("results"):
                os.makedirs("results")
            dictionary_knee = {"number": marker_frequency_total["number"], "cluster": list_labels_DBSCAN_knee}
            data_frame_knee = pd.DataFrame(dictionary_knee)
            data_frame_knee.to_csv("results/trajectory_clusters_DBSCAN_knee_neighbors_" + str(neighbors) + "_window_" + str(window) + ".csv", index = False)
        
        # DBSCAN using the elbow point

        if elbow_point_dist:
            print("Elbow:", elbow_point, elbow_point_dist)
            time_now = time()
            DBSCAN_elbow = DBSCAN(min_samples = neighbors, eps = elbow_point_dist).fit(X)
            time_dict_clustering_DBSCAN["time"].append(time() - time_now)
            time_dict_clustering_DBSCAN["task"].append("DBSCAN_elbow_neighbors_" + str(neighbors) + "_window_" + str(window))
            list_labels_DBSCAN_elbow = list(DBSCAN_elbow.labels_)

            print("DBSCAN clustering time (elbow):", time_dict_clustering_DBSCAN["time"][-1])

            label_set_elbow = set(list_labels_DBSCAN_elbow)
            number_of_clusters_elbow = len(label_set_elbow)
            print("DBSCAN clusters (elbow):", number_of_clusters_elbow)
            for cluster_marker in sorted(list(label_set_elbow)):
                print("Cluster marker:", cluster_marker, "Cluster size:", list_labels_DBSCAN_elbow.count(cluster_marker))

            prepare_figure()
            plt.title("DBSCAN clusters (elbow)\nneighbors: " + str(neighbors) + ", window: " + str(window))
            for cluster_marker in sorted(list(label_set_elbow)):
                x_values_PCA = [new_data_PCA["PCA_feature_1"][ix] for ix in range(len(list_labels_DBSCAN_elbow)) if list_labels_DBSCAN_elbow[ix] == cluster_marker]
                y_values_PCA = [new_data_PCA["PCA_feature_2"][ix] for ix in range(len(list_labels_DBSCAN_elbow)) if list_labels_DBSCAN_elbow[ix] == cluster_marker]
                plt.scatter(x_values_PCA, y_values_PCA, label = str(cluster_marker + 1), zorder = cluster_marker + 3)
            plt.xlabel("PCA feature 1")
            plt.ylabel("PCA feature 2")
            plt.legend()
            if not os.path.isdir("plots/PCA/DBSCAN/elbow"):
                os.makedirs("plots/PCA/DBSCAN/elbow")
            plt.savefig("plots/PCA/DBSCAN/elbow/PCA_DBSCAN_elbow_neighbors_" + str(neighbors) + "_window_" + str(window) + ".pdf", bbox_inches = "tight")
            plt.savefig("plots/PCA/DBSCAN/elbow/PCA_DBSCAN_elbow_neighbors_" + str(neighbors) + "_window_" + str(window) + ".svg", bbox_inches = "tight")
            plt.savefig("plots/PCA/DBSCAN/elbow/PCA_DBSCAN_elbow_neighbors_" + str(neighbors) + "_window_" + str(window) + ".png", bbox_inches = "tight")
            plt.close()

            prepare_figure()
            plt.title("DBSCAN clusters (elbow)\nneighbors: " + str(neighbors) + ", window: " + str(window))
            for cluster_marker in sorted(list(label_set_elbow)):
                x_values_TSNE = [new_data_TSNE["TSNE_feature_1"][ix] for ix in range(len(list_labels_DBSCAN_elbow)) if list_labels_DBSCAN_elbow[ix] == cluster_marker]
                y_values_TSNE = [new_data_TSNE["TSNE_feature_2"][ix] for ix in range(len(list_labels_DBSCAN_elbow)) if list_labels_DBSCAN_elbow[ix] == cluster_marker]
                plt.scatter(x_values_TSNE, y_values_TSNE, label = str(cluster_marker + 1), zorder = cluster_marker + 3)
            plt.xlabel("TSNE feature 1")
            plt.ylabel("TSNE feature 2")
            plt.legend()
            if not os.path.isdir("plots/TSNE/DBSCAN/elbow"):
                os.makedirs("plots/TSNE/DBSCAN/elbow")
            plt.savefig("plots/TSNE/DBSCAN/elbow/TSNE_DBSCAN_elbow_neighbors_" + str(neighbors) + "_window_" + str(window) + ".pdf", bbox_inches = "tight")
            plt.savefig("plots/TSNE/DBSCAN/elbow/TSNE_DBSCAN_elbow_neighbors_" + str(neighbors) + "_window_" + str(window) + ".svg", bbox_inches = "tight")
            plt.savefig("plots/TSNE/DBSCAN/elbow/TSNE_DBSCAN_elbow_neighbors_" + str(neighbors) + "_window_" + str(window) + ".png", bbox_inches = "tight")
            plt.close()
            
            if not os.path.isdir("results"):
                os.makedirs("results")
            dictionary_elbow = {"number": marker_frequency_total["number"], "cluster": list_labels_DBSCAN_elbow}
            data_frame_elbow = pd.DataFrame(dictionary_elbow)
            data_frame_elbow.to_csv("results/trajectory_clusters_DBSCAN_elbow_neighbors_" + str(neighbors) + "_window_" + str(window) + ".csv", index = False)

        if not os.path.isdir("results"):
            os.makedirs("results")
        data_frame_time_clustering_DBSCAN = pd.DataFrame(time_dict_clustering_DBSCAN)
        data_frame_time_clustering_DBSCAN.to_csv("results/time_clustering_DBSCAN_neighbors_" + str(neighbors) + "_window_" + str(window) + ".csv", index = False)
        
    # KMeans clustering

    clusters = 0
    try:
        clusters = int(args.clusters)
    except:
        print("The number of clusters for KMeans clustering (" + str(args.clusters) + ") is not an integer greater than 2, KMeans clustering is not applied.")
        clusters = 0

    if clusters < 2:
        print("The number of clusters for KMeans clustering (" + str(args.clusters) + ") is not an integer greater than 2, KMeans clustering is not applied.")
        clusters = 0

    if clusters:
        time_dict_clustering_KMeans = {"task": [], "time": []}
        time_now = time()
        KMeans_clusters = KMeans(n_clusters = clusters, random_state = 42, init = "k-means++", n_init = "auto").fit(X)
        time_dict_clustering_KMeans["time"].append(time() - time_now)
        time_dict_clustering_KMeans["task"].append("KMeans_clusters_" + str(clusters) + "_window_" + str(window))
        
        print("KMeans clustering time:", time_dict_clustering_KMeans["time"][-1])

        if not os.path.isdir("results"):
            os.makedirs("results")
        data_frame_time_clustering_KMeans = pd.DataFrame(time_dict_clustering_KMeans)
        data_frame_time_clustering_KMeans.to_csv("results/time_clustering_KMeans_clusters_" + str(clusters) + "_window_" + str(window) + ".csv", index = False)
        
        label_list_KMeans = list(KMeans_clusters.labels_) 
        label_set_KMeans = set(label_list_KMeans)
        number_of_clusters_KMeans = len(label_set_KMeans)
        print("KMeans clusters:", number_of_clusters_KMeans)
        
        for cluster_marker in sorted(list(label_set_KMeans)):
            print("Cluster marker:", cluster_marker, "Cluster size:", label_list_KMeans.count(cluster_marker))

            prepare_figure()
            plt.title("KMeans clusters\nclusters: " + str(clusters) + ", window: " + str(window))
            for cluster_marker in sorted(list(label_set_KMeans)):
                x_values_PCA = [new_data_PCA["PCA_feature_1"][ix] for ix in range(len(label_list_KMeans)) if label_list_KMeans[ix] == cluster_marker]
                y_values_PCA = [new_data_PCA["PCA_feature_2"][ix] for ix in range(len(label_list_KMeans)) if label_list_KMeans[ix] == cluster_marker]
                plt.scatter(x_values_PCA, y_values_PCA, label = str(cluster_marker + 1), zorder = cluster_marker + 2)
            plt.xlabel("PCA feature 1")
            plt.ylabel("PCA feature 2")
            plt.legend()
            if not os.path.isdir("plots/PCA/KMeans"):
                os.makedirs("plots/PCA/KMeans")
            plt.savefig("plots/PCA/KMeans/PCA_KMeans_clusters_" + str(clusters) + "_window_" + str(window) + ".pdf", bbox_inches = "tight")
            plt.savefig("plots/PCA/KMeans/PCA_KMeans_clusters_" + str(clusters) + "_window_" + str(window) + ".svg", bbox_inches = "tight")
            plt.savefig("plots/PCA/KMeans/PCA_KMeans_clusters_" + str(clusters) + "_window_" + str(window) + ".png", bbox_inches = "tight")
            plt.close()

            prepare_figure()
            plt.title("KMeans clusters\nclusters: " + str(clusters) + ", window: " + str(window))
            for cluster_marker in sorted(list(label_set_KMeans)):
                x_values_TSNE = [new_data_TSNE["TSNE_feature_1"][ix] for ix in range(len(label_list_KMeans)) if label_list_KMeans[ix] == cluster_marker]
                y_values_TSNE = [new_data_TSNE["TSNE_feature_2"][ix] for ix in range(len(label_list_KMeans)) if label_list_KMeans[ix] == cluster_marker]
                plt.scatter(x_values_TSNE, y_values_TSNE, label = str(cluster_marker + 1), zorder = cluster_marker + 2)
            plt.xlabel("TSNE feature 1")
            plt.ylabel("TSNE feature 2")
            plt.legend()
            if not os.path.isdir("plots/TSNE/KMeans"):
                os.makedirs("plots/TSNE/KMeans")
            plt.savefig("plots/TSNE/KMeans/TSNE_KMeans_clusters_" + str(clusters) + "_window_" + str(window) + ".pdf", bbox_inches = "tight")
            plt.savefig("plots/TSNE/KMeans/TSNE_KMeans_clusters_" + str(clusters) + "_window_" + str(window) + ".svg", bbox_inches = "tight")
            plt.savefig("plots/TSNE/KMeans/TSNE_KMeans_clusters_" + str(clusters) + "_window_" + str(window) + ".png", bbox_inches = "tight")
            plt.close()

        if not os.path.isdir("results"):
            os.makedirs("results")
        dictionary_KMeans = {"number": marker_frequency_total["number"], "cluster": label_list_KMeans}
        data_frame_KMeans = pd.DataFrame(dictionary_KMeans)
        data_frame_KMeans.to_csv("results/trajectory_clusters_KMeans_clusters_" + str(clusters) + "_window_" + str(window) + ".csv", index = False)