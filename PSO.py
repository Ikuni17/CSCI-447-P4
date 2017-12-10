import random
import numpy as np
import KM

class Particle:
    gbest = None
    v1 = None
    v2 = None

    def __init__(self, clusters, variables):
        self.pbest = None
        self.clusterPositions = []
        self.velocity = []
        for i in range(clusters):
            self.clusterPositions.append([])
            self.velocity.append([])
            for j in range(variables):
                self.clusterPositions[i].append(random.randint(0, 30))
                self.velocity.append(0)
        self.clusterPositions = np.array(self.clusterPositions)
        self.evaluate()

    # calculate the distance of the clusters and update gbest and pbest
    def evaluate(self, data):
        clusters = KM.associate_data(clusterPositions, data)

        # TODO: Calculate the performancehere
        averageDist = 0

        # update personal and global bests if we found a better position
        if averageDist < Particle.gbest:
            Particle.gbest = averageDist
            Particle.pbest = averageDist
        elif averageDist < self.pbest:
            Particle.pbest = averageDist

    def calcVelocity(self):
        for i in range(len(velocity)):
            for j in range(len(velocity[i])):
                self.velocity[i][j] = self.velocity[i][j] + v1*random.uniform(0, 1)*(self.pbest[i][j] - self.clusterPosition[i][j]) + Particle.v2*random.uniform(0, 1)*(Particle.gbest[i][j] - self.clusterPosition[i][j])
        return self.velocity

    def move(self):
        np.add(self.clusterPositions, calcVelocity())

class PSO:
    def __init__(self, numParticles, clusters, data, v1 = 0.1, v2 = 0.1):
        #initialize the particles with random values
        self.particles = self.initSwarm(numParticles, clusters, len(data[0]), v1, v2)
        self.data = data

    def runSwarm(self,rounds):
        for i in range(rounds):
            for particle in self.particles:
                particle.move()
                particle.evaluate(self.data)

    def initSwarm(self, numParticles, clusters, variables, v1, v2):
        particles = []
        for i in range(numParticles):
            particles.append(Particle(clusters, variables))
        Particle.v1 = v1
        Particle.v2 = v2
        return particles

if __name__ == '__main__':
    data = [[1,5, 7], [5,19, 12], [14, 4, 7], [5, 18, 21], [9,41, 25], [13, 32, 15]]
    pso = PSO(10, 2, data)
    pso.runSwarm(100)
