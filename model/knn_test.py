import numpy as np
from lib.pointops.functions.pointops import KNNQuery
import time

batch_size = 8
num_points = 819200
K = 8
pc = np.random.rand(batch_size, num_points, 3).astype(np.float32)

# nearest neighbours
start = time.time()
neigh_idx = KNNQuery(pc, pc, K, omp=True)
print(neigh_idx.__sizeof__())
