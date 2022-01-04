import numpy as np

cities = 10
Graph = np.zeros((cities,2))
Graph[:,0] = np.random.rand(cities)*cities
Graph[:,1] = np.random.rand(cities)*cities

with open('Graph_bigger.txt', 'w') as file_out:
    for row in range(len(Graph)):
        for col in range(0,2):
            file_out.write(str(Graph[row,col]))
            if col < 1:
                file_out.write(', ')
        file_out.write('\n')
