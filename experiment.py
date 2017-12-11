'''
CSCI 447: Project 4
Group 28: Trent Baker, Logan Bonney, Bradley White
December 9, 2017
'''

import ACO
import CL
import DB
import KM
import PSO
import pandas
import matplotlib.pyplot as plt
import numpy as np
import random

''' This program runs each clustering algorithm on a specific dataset, and then evaluates the resulting clusters. Also,
it has some auxiliary functions such as reading in datasets, graphing 2D data, and testing algorithms.
'''


# Read in a csv dataset, convert all values to numbers, and return as a 2D list
def get_dataset(csv_path):
    df = pandas.read_csv(csv_path, header=None)
    return df.values.tolist()


# Create a dictionary of all datasets in csv_names
def load_datasets(csv_names):
    # Dictionary with dataset name as the key and a 2D list of vectors as the value
    datasets = {}

    # Populate the dictionary using helper function
    for i in range(len(csv_names)):
        datasets[csv_names[i]] = get_dataset('datasets\\{0}.csv'.format(csv_names[i]))

    return datasets


# Generate random, clustered data for testing and output to csv
def gen_data(mu, sigma, cluster_size, magnitude, dimension=2):
    if len(mu) != len(sigma):
        print('Invalid data generation parameters')
    else:
        data = []
        for i in range(len(mu)):
            cluster = np.ndarray.tolist(sigma[i] * np.random.randn(cluster_size, dimension) + mu[i] * magnitude)
            for point in cluster:
                data.append(point)
        # Shuffle the data so its not clustered already
        random.shuffle(data)
        # Write to CSV for later use if needed
        df = pandas.DataFrame(data)
        df.to_csv('data.csv', index=False)
        return data


# Helper function to test CL
def process_pairs(data):
    x = []
    y = []
    for point in data:
        x.append(point[0])
        y.append(point[1])
    return x, y


# Used to tune CL parameters
def test_CL():
    cluster_size = 10000
    num_clusters = 5
    epsilon_step = .0000001
    magnitude = 10

    path = 'datasets\\bean.csv'
    data = get_dataset(path)
    print('CL-[' + str(num_clusters) + ', ' + str(epsilon_step) + ', ' + path + '].png')

    # Run CL then graph the clusters
    clusters = CL.train(data, num_clusters, epsilon_step)
    for key in clusters.keys():
        x, y = process_pairs(clusters[key])
        plt.scatter(x, y, linestyle='None', marker=".")

    print(str(cluster_size) + '\n' + str(num_clusters) + '\n' + str(epsilon_step) + '\n' + str(magnitude))

    # Used old evaluate function
    # print('evaluate result: ' + str(evaluate_clusters(dict_to_list(clusters))))
    plt.title('CL: num_clusters = ' + str(num_clusters) + ', epsilon_step = ' + str(epsilon_step))
    plt.show()


# Converts a dictionary of clusters to a 2D matrix for analysis
def dict_to_list(dict):
    list = []
    for key in dict.keys():
        list.append(dict[key])
    return list


# Main experiment that runs each algorithm on each dataset, then evaluate the resulting clusters
def main():
    csv_names = ['airfoil', 'concrete', 'forestfires', 'machine', 'yacht']
    # Precomputed minimum for DBScan to save computations and user interaction
    db_mins = [9, 75, 88, 2000, 3]
    datasets = load_datasets(csv_names)
    # Number of clusters for CL, KM and PSO
    num_clusters = 5
    # Number of particles for PSO
    num_particles = 10

    for i in range(len(csv_names)):
        print("Starting ACO with:", csv_names[i])
        aco = ACO.ACO(data=datasets[csv_names[i]])
        clusters = dict_to_list(aco.main(csv_names[i], max_iter=1000000))
        evaluate_clusters('ACO', csv_names[i], clusters)

        print("Starting CL with:", csv_names[i])
        clusters = dict_to_list(CL.train(datasets[csv_names[i]], num_clusters))
        evaluate_clusters('CL', csv_names[i], clusters)

        print("Starting DB with:", csv_names[i])
        clusters = DB.DBScan(datasets[csv_names[i]], db_mins[i])
        evaluate_clusters('DB', csv_names[i], clusters)

        print("Starting KM with:", csv_names[i])
        clusters = dict_to_list(KM.train(datasets[csv_names[i]], num_clusters))
        evaluate_clusters('KM', csv_names[i], clusters)

        print("Starting PSO with:", csv_names[i])
        pso = PSO.PSO(num_particles, num_clusters, datasets[csv_names[i]])
        clusters = pso.runSwarm(100)
        evaluate_clusters('PSO', csv_names[i], clusters)


# Helper method to graph clusters during tuning
def graph2dClusters(data):
    for cluster in data:
        xVal = [x[0] for x in cluster]
        yVal = [y[1] for y in cluster]
        plt.scatter(xVal, yVal, linestyle='None', marker=".")
        # plt.scatter(sum(xVal)/len(xVal), sum(yVal)/len(yVal))
    plt.show()


'''Analyze the clusters returned by an algorithm. Finds the average distance to the center for the points in a cluster,
the average points per cluster, the average distance between centers of the clusters, and number of clusters. Then
writes the results to file.
'''
def evaluate_clusters(algorithm, dataset, clusters):
    # Each cluster is in a vector
    amount_clusters = len(clusters)
    centers = []
    average_dist = 0
    num_points = 0

    # Iterate through all clusters
    for cluster in clusters:
        # Get the center and add to the sum of points
        center_point = [0] * len(cluster[0])
        num_points = num_points + len(cluster)

        # Calculate the average in each dimension for the cluster
        for i in range(len(center_point)):
            dimCut = [dim[i] for dim in cluster]
            center_point[i] = sum(dimCut) / len(dimCut)
        # Add to the list of centers
        centers.append(center_point)

        # Calculate the distance from each point to its center
        for point in cluster:
            average_dist = average_dist + KM.euclidian_distance(point, center_point)

    # Get the average distance for all points to the center, represents how tight clusters are
    average_dist = average_dist / num_points
    # Get the average points per cluster
    average_pts = num_points / amount_clusters

    center_dist = 0

    for x in centers:
        for y in centers:
            center_dist += KM.euclidian_distance(x, y)

    # Add a factor of 2 since each distance is found twice
    center_dist /= (len(centers) * 2)

    # Append to the results file
    with open('Results.txt', "a") as output:
        output.write("{0},{1},{2},{3},{4},{5}\n".format(
            algorithm, dataset, amount_clusters, average_pts, average_dist, center_dist))


if __name__ == '__main__':
    main()
