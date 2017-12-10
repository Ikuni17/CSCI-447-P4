'''
CSCI 447: Project 4
Group 28: Trent Baker, Logan Bonney, Bradley White
December 9, 2017
'''

import ACO
import pandas


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


if __name__ == '__main__':
    main()