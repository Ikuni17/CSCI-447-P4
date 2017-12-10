'''
CSCI 447: Project 4
Group 28: Trent Baker, Logan Bonney, Bradley White
December 9, 2017
'''

import ACO
import pandas
import matplotlib.pyplot as plt
import numpy as np

# Read in a csv dataset, convert all values to numbers, and return as a 2D list
def get_dataset(csv_path):
    df = pandas.read_csv(csv_path, header=None)
    return df.values.tolist()


def load_datasets():
    # Dataset names
    #csv_names = ['airfoil', 'concrete', 'forestfires', 'machine', 'yacht']
    csv_names = ['machine']
    # Dictionary with dataset name as the key and a 2D list of vectors as the value
    datasets = {}

    # Populate the dictionary using helper function
    for i in range(len(csv_names)):
        datasets[csv_names[i]] = get_dataset('datasets\\{0}.csv'.format(csv_names[i]))

    return datasets

def gen_data():
    return np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])), (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])), (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))


def main():
    datasets = load_datasets()
    aco = ACO.ACO(data=datasets['machine'])
    aco.main(max_iter=10000)
    '''aco = ACO.ACO(data=datasets['machine'], gammas=[1, 0.1, 0.5])
    aco.main(max_iter=100000)
    aco = ACO.ACO(data=datasets['machine'], gammas=[1, 0.5, 0.5])
    aco.main(max_iter=100000)
    aco = ACO.ACO(data=datasets['machine'], gammas=[1, 0.7, 0.5])
    aco.main(max_iter=100000)
    aco = ACO.ACO(data=datasets['machine'], gammas=[1, 0.9, 0.5])
    aco.main(max_iter=100000)
    aco = ACO.ACO(data=datasets['machine'], gammas=[1, 0.1, 0.1])
    aco.main(max_iter=100000)
    aco = ACO.ACO(data=datasets['machine'], gammas=[1, 0.1, 0.3])
    aco.main(max_iter=100000)
    aco = ACO.ACO(data=datasets['machine'], gammas=[1, 0.1, 0.7])
    aco.main(max_iter=100000)
    aco = ACO.ACO(data=datasets['machine'], gammas=[1, 0.1, 0.9])
    aco.main(max_iter=100000)'''

def graph2dClusters(data):
    for cluster in data:
        xVal = [x[0] for x in cluster]
        yVal = [y[1] for y in cluster]
        plt.scatter(xVal, yVal, linestyle='None', marker = ".")
    plt.show()

if __name__ == '__main__':
    main()
