'''
CSCI 447: Project 4
Group 28: Trent Baker, Logan Bonney, Bradley White
December 7, 2017
'''

import random
from collections import OrderedDict


class Ant():
    def __init__(self, location):
        self.location = location
        self.carrying = None

    def move(self, step):
        pass

    def pickup(self):
        pass

    def drop(self):
        pass


class Datum():
    def __init__(self, data):
        self.input_vector = data
        # TODO get class from last? index of vector
        self.classification = None

    def similarity(self, point):
        pass


class Grid():
    def __init__(self, dimension, data, patch_size=1, gammas=[1, 0.01, 0.01]):
        self.grid = None
        self.patch_size = patch_size
        self.gammas = gammas

    def density(self, ant):
        pass

    def prob_pick(self, ant):
        pass

    def prob_drop(self, ant):
        pass


class ACO():
    def __init__(self, data, patch_size=1, step_size=1):
        length = len(data)
        self.grid_size = length * 10
        # OrderedDict where the keys are tuples for coordinates (x, y), and the value is a dictionary
        self.grid = self.init_grid(self.grid_size)
        self.ants = self.init_ants(length * 2)
        self.step_size = step_size

    def eval_clusters(self):
        pass

    def init_grid(self, dim):
        grid = OrderedDict()

        for x in range(dim):
            for y in range(dim):
                grid[(x, y)] = {'Ant': None, 'Datum': None}

        return grid

    def init_ants(self, num_ants):
        ants = []
        # TODO While loop until an open position is found
        for i in range(num_ants):
            x = random.randint(0, self.grid_size)
            y = random.randint(0, self.grid_size)
            print("x:", x)
            print("y:", y)

    def main(self, max_iter=10000):
        pass


if __name__ == '__main__':
    aco = ACO(data=[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    print(aco.grid.keys())
    print(aco.grid[(25, 25)])
