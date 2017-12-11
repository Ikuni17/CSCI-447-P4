import CL
import KM
import experiment
import pandas
import copy
import random
import matplotlib.pyplot as plt

min_neighbors = 4


def DBScan(data, min_distance):
    labels = [None] * len(data)
    clusters = []
    cluster_id = 0
    for point in range(len(data)):
        cluster = []
        points_to_process = [point]
        while points_to_process:
            current_point = points_to_process.pop(0)
            neighbors = get_neighbors(data[current_point], data, min_distance)
            if labels[current_point] == None:
                if len(neighbors) > 4:
                    labels[current_point] = cluster_id
                    cluster.append(data[current_point])
                    points_to_process = points_to_process + neighbors
                elif len(neighbors) > 0:
                    for neighbor in neighbors:
                        if labels[neighbor] != None and labels[neighbor] == cluster_id:
                            labels[current_point] = cluster_id
                            cluster.append(data[current_point])

            cluster_id += 1
        if cluster:
            clusters.append(cluster)

    noise = []
    for i in range(len(labels)):
        if labels[i] == None:
            noise.append(data[i])
            labels[i] = cluster_id
    print(noise)
    cluster.append(noise)
    return clusters


def expand_cluster(point, data, labels, min_distance, cluster_id, cluster):
    neighbors = get_neighbors(data[point], data, min_distance)
    if len(neighbors) > 4:
        labels[point] = cluster_id
        cluster.append(data[point])
        for neighbor in neighbors:
            if labels[neighbor] == None:
                expand_cluster(neighbor, data, labels, min_distance, cluster_id, cluster)
    elif len(neighbors) > 0:
        for neighbor in neighbors:
            if labels[neighbor] != None and labels[neighbor] == cluster_id:
                labels[point] = cluster_id
                cluster.append(data[point])


def calc_distance_to_kth(data, sample_size, k):
    sample = []
    data = copy.deepcopy(data)

    if sample_size > len(data):
        sample_size = len(data)

    for i in range(sample_size):
        sample.append(data.pop(int(random.random() * sample_size)))
        sample_size -= 1

    kth_distances = []
    for point in sample:
        distances = []
        for other_point in sample:
            distances.append(KM.euclidian_distance(point, other_point))
        distances.sort()
        kth_distances.append(distances[k - 1])

    kth_distances = list(filter(lambda a: a != 0.0, kth_distances))
    kth_distances = list(filter(lambda a: a != None, kth_distances))
    # for i in range(500):
    # print(distances[i])
    # return distances[k]
    kth_distances.sort()
    plt.plot(kth_distances)
    plt.show()

    return float(input('Please enter the y value where exponential increase begins: '))


def get_neighbors(point, data, min_distance):
    neighbors = []
    for i in range(len(data)):
        if KM.euclidian_distance(point, data[i]) < min_distance:
            neighbors.append(i)
    return neighbors


def read_data(path):
    df = pandas.read_csv(path, header=None)
    return df.values.tolist()


if __name__ == '__main__':
    data = read_data('datasets/machine.csv')
    max_data_size = 6000
    if len(data) > max_data_size:
        new_data = []
        for i in range(max_data_size):
            new_data.append(data.pop(int(random.random() * len(data))))
    sample_size = 5000
    k = 4
    min_distance = calc_distance_to_kth(data, sample_size, k)
    # min_distance = 197
    # print(str(min_distance))
    dbscan = DBScan(data, min_distance)
    experiment.evaluate_clusters('DB','airfoil',dbscan)
    #print('Average distance to center of cluster: {0}, number of clusters: {1}'.format(
    #    experiment.evaluate_clusters(dbscan), len(dbscan)))
    #experiment.graph2dClusters(dbscan)
# print(str(dbscan))
# for i in dbscan:
#	print('cluster:')
#	for point in i:
#		print(point)
