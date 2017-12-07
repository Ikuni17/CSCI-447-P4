'''
CSCI 447: Project 4
Group 28: Trent Baker, Logan Bonney, Bradley White
December 7, 2017
'''


class Ant():
    def __init__(self, x, y):
        self.location = (x, y)
        self.carrying = None

    def move(self, step):
        pass

    def pickup(self):
        pass

    def drop(self):
        pass


class Point():
    def __init__(self, data):
        self.input_vector = data

    def similarity(self, point):
        pass


class Grid():
    def __init__(self, dimension, data, patch_size, gammas):
        self.dimension = dimension
        self.grid = None
        self.patch_size = patch_size
        self.gammas = gammas

    def density(self, ant):
        pass

    def prob_pick(self, ant):
        pass

    def prob_drop(self, ant):
        pass

    def eval_clusters(self):
        pass

class ACO():
    def __int__(self, max_iter, num_ants, step_size):
        self.max_iter = max_iter
        self.num_ants = num_ants
        self.grid = None
        self.ants = None
        self.step_size = step_size

    def main(self):
        pass

if __name__ == '__main__':
    print("Not implemented")