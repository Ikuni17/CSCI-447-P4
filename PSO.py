import random
import numpy as np
import KM
import experiment


class Particle:
    gbest = None
    gscore = None
    v1 = None
    v2 = None

    def __init__(self, clusters, dimensions):
        self.pbest = None
        self.pscore = None
        self.centers_pos = []
        self.velocity = []
        # Create an array for each center
        for i in range(clusters):
            self.centers_pos.append([])
            self.velocity.append([])
            for j in range(dimensions):
                self.centers_pos[i].append(random.uniform(0, 10))
                self.velocity[i].append(0)
        self.centers_pos = self.centers_pos

    # calculate the distance of the clusters and update gbest and pbest
    def evaluate(self, data):
        clusters = Particle.get_clusters(self.centers_pos, data)
        performance = experiment.evaluate_clusters(clusters)
        return performance

    def get_clusters(centers, data):
        clusters = []
        for i in range(len(centers)):
            clusters.append([])

        for point in data:
            closest = 0
            for i in range(len(centers)):
                if KM.euclidian_distance(centers[i], point) < KM.euclidian_distance(centers[closest], point):
                    closest = i
            clusters[closest].append(point)
        return clusters

    def update_velocity(self):
        for i in range(len(self.velocity)):
            for j  in range(len(self.velocity[i])):
                self.velocity[i][j] = self.velocity[i][j] + Particle.v1 * random.uniform(0, 1) * (self.pbest[i][j] - self.centers_pos[i][j]) + Particle.v2 * random.uniform(0, 1) * (Particle.gbest[i][j] - self.centers_pos[i][j])

    def move(self, data):
        self.update_velocity()
        for i in range(len(self.centers_pos)):
            for j in range(len(self.centers_pos[i])):
                self.centers_pos[i][j] = self.centers_pos[i][j] + self.velocity[i][j]
        performance = self.evaluate(data)
        # update personal and global bests if we found a better position
        if performance < Particle.gscore:
            Particle.gbest = self.centers_pos
            Particle.pbest = self.centers_pos
        elif performance < self.pscore:
            Particle.pbest = self.centers_pos

class PSO:
    def __init__(self, numParticles, clusters, data, v1=0.01, v2=0.01):
        # initialize the particles with random values
        self.particles = self.initSwarm(numParticles, clusters, len(data[0]), v1, v2)
        self.data = data

    def runSwarm(self, rounds):
        for i in range(rounds):
            for particle in self.particles:
                particle.move(self.data)
        print('calculaing clusters')
        return Particle.get_clusters(Particle.gbest, self.data)

    def initSwarm(self, numParticles, clusters, variables, v1, v2):
        particles = []
        for i in range(numParticles):
            particles.append(Particle(clusters, variables))
            particles[i].pbest = particles[i].centers_pos
            particles[i].pscore = particles[i].evaluate(data)
        Particle.gbest = particles[0].pbest
        Particle.gscore = particles[0].pscore
        Particle.v1 = v1
        Particle.v2 = v2
        return particles

if __name__ == '__main__':
    data_name= '3_clusters.csv'
    iterations = 10
    data = KM.read_data(data_name)
    clusters = 3
    particles = 10
    pso = PSO(particles, clusters, data)
    print('Running PSO on {0} with {1} clusters and {2} particles for {3} iterations'.format(data_name, clusters, particles, iterations))
    best = pso.runSwarm(iterations)
    print(Particle.gbest)
    for cluster in best:
        print('cluster: {0}'.format(cluster))

    print(experiment.evaluate_clusters(best))
    experiment.graph2dClusters(best)
