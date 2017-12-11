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
import time


# Read in a csv dataset, convert all values to numbers, and return as a 2D list
def get_dataset(csv_path):
    df = pandas.read_csv(csv_path, header=None)
    return df.values.tolist()


def load_datasets(csv_names):
    # Dictionary with dataset name as the key and a 2D list of vectors as the value
    datasets = {}

    # Populate the dictionary using helper function
    for i in range(len(csv_names)):
        datasets[csv_names[i]] = get_dataset('datasets\\{0}.csv'.format(csv_names[i]))

    return datasets


def gen_data(mu, sigma, cluster_size, magnitude, dimension=2):
    if len(mu) != len(sigma):
        print('invalid data generation parameters')
    else:
        data = []
        for i in range(len(mu)):
            # print('cluster')
            cluster = np.ndarray.tolist(sigma[i] * np.random.randn(cluster_size, dimension) + mu[i] * magnitude)
            # print(str(cluster))
            for point in cluster:
                data.append(point)
        random.shuffle(data)
        # KM.print_vectors('Data:', data)
        df = pandas.DataFrame(data)
        df.to_csv('data.csv', index=False)
        return data


def process_pairs(data):
    x = []
    y = []
    for point in data:
        x.append(point[0])
        y.append(point[1])
    return x, y


def test_CL():
    cluster_size = 10000
    num_clusters = 5
    epsilon_step = .0000001
    magnitude = 10

    # mu = []
    # sigma = []
    # for i in range(num_clusters):
    #     mu.append(random.random() * magnitude)
    #     sigma.append(random.random() * magnitude)
    # data = gen_data(mu, sigma, cluster_size, magnitude)
    # print('CL-[' + str(num_clusters) + ', ' + str(epsilon_step) + '].png')

    path = 'bean'
    data = get_dataset(path + '.csv')
    print('CL-[' + str(num_clusters) + ', ' + str(epsilon_step) + ', ' + path + '].png')

    clusters = CL.train(data, num_clusters, epsilon_step)
    for key in clusters.keys():
        x, y = process_pairs(clusters[key])
        plt.scatter(x, y, linestyle='None', marker=".")

    print(str(cluster_size) + '\n' + str(num_clusters) + '\n' + str(epsilon_step) + '\n' + str(magnitude))

    print('evaluate result: ' + str(evaluate_clusters(dict_to_list(clusters))))
    plt.title('CL: num_clusters = ' + str(num_clusters) + ', epsilon_step = ' + str(epsilon_step))
    plt.show()


def dict_to_list(dict):
    list = []
    for key in dict.keys():
        list.append(dict[key])
    return list


def main():
    # test_CL()
    # print(str(clusters))

    # plt.scatter(x, y)
    # plt.scatter(y, x)
    # plt.show()

    csv_names = ['airfoil', 'concrete', 'forestfires', 'machine', 'yacht']
    db_mins = [9, 75, 88, 2000, 3]
    datasets = load_datasets(csv_names)
    num_clusters = 5

    for i in range(len(csv_names)):
        '''print("Starting ACO with:", csv_names[i])
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
        evaluate_clusters('KM', csv_names[i], clusters)'''

        print("Starting PSO with:", csv_names[i])
        pso = PSO.PSO(10, num_clusters, datasets[csv_names[i]])
        clusters = pso.runSwarm(100)
        evaluate_clusters('PSO', csv_names[i], clusters)


# TODO Remove
def test_KM(datasets, csv_names):
    results = {}
    for name in csv_names:
        results[name] = {2: 0, 3: 0, 4: 0, 5: 0}
        for k in range(2, 6):
            print("Starting test with k={0} on {1}".format(k, name))
            for i in range(25):
                clusters = KM.train(datasets[name], k)
                for key in clusters.keys():
                    if len(clusters[key]) == 0:
                        results[name][k] += 1

    print(results)


def graph2dClusters(data):
    for cluster in data:
        xVal = [x[0] for x in cluster]
        yVal = [y[1] for y in cluster]
        plt.scatter(xVal, yVal, linestyle='None', marker=".")

    plt.show()


def evaluate_clusters(algorithm, dataset, clusters):
    amount_clusters = len(clusters)
    centers = []
    average_dist = 0
    num_points = 0
    for cluster in clusters:
        if(cluster):
            center_point = [0]*len(cluster[0])
            num_points = num_points + len(cluster)

            # Calculate the average in each dimension for the cluster
            for i in range(len(center_point)):
                dimCut = [dim[i] for dim in cluster]
                center_point[i] = sum(dimCut)/len(dimCut)
            centers.append(center_point)

        # Calculate the distance from each point to its center
        for point in cluster:
            average_dist = average_dist + KM.euclidian_distance(point, center_point)

    average_dist = average_dist / num_points
    average_pts = num_points / amount_clusters

    center_dist = 0

    for x in centers:
        for y in centers:
            center_dist += KM.euclidian_distance(x, y)

    center_dist /= len(centers)

    with open('Results.txt', "a") as output:
        output.write(
            "{0},{1},{2},{3},{4},{5},{6}\n".format(algorithm, dataset, amount_clusters, average_pts, average_dist,
                                                 center_dist, time.ctime(time.time())))


if __name__ == '__main__':
    main()
