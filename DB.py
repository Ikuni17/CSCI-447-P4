import CL
import KM
import pandas

minNeighbors = 4

def DBScan(data, minDist):
    clusters = []
    for point in data:
        clusters.append(getGroup(point, list(data), minDist))
    return clusters

def getGroup(point, data, minDist):
    group = []

    for dataPoint in data:
        distance = KM.euclidian_distance(point, dataPoint)
        if distance < minDist and distance > 0:
            group.append(dataPoint)
            data.remove(dataPoint)

    if len(group) > 4:
        for groupMember in group:
            group + getGroup(dataPoint, data, minDist)
    else:
        return group

    return group


def read_data(path):
    df = pandas.read_csv(path, header=None)
    return df.values.tolist()


if __name__ == '__main__':
    data = read_data('datasets/yacht.csv')
    minDist = 30
    dbscan = DBScan(data, minDist)
    for i in dbscan:
        print('cluster:')
        for point in i:
            print(point)
