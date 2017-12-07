'''
CSCI 447: Project 4
Group 28: Trent Baker, Logan Bonney, Bradley White
December 7, 2017
'''


class Ant():
    def __init__(self, max_dim):
        self.location = None
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
    def __int__(self, data, patch_size=1, step_size=1):
        length = len(data)
        grid_size = length * 10
        self.grid = Grid(grid_size, data)
        self.ants = [Ant(grid_size) for x in range(length * 2)]
        self.step_size = step_size

    def eval_clusters(self):
        pass

    def main(self, max_iter=10000):
        pass


if __name__ == '__main__':
    print("Not implemented")
