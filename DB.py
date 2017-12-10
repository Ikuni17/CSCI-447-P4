import KM

minNeighbors = 4

def DBScan(data, minDist):
    clusters = []
    for point in data:
        clsuters.append(point.getGroup(point, list(data), minDist))

def getGroup(point, data, minDist):
    group = []

    for dataPoint in data:
        distance = KM.euclidian_distance(point, dataPoint)
        if distance < minDist and distance > 0:
            group.append(datapoint)
            data.remove(dataPoint)

    if(len(group) > 4:
        for groupMember in group:
            group + getGroup(dataPoint, data, minDist)
    else:
        return group

    return neighbors

if __name__ == '__main__':
    dbscan = DBScan()
