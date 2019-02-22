import numpy as np
class Map(object):
    def __init__(self):
        '''
        Initialize Map
        Output:
            Map - Object resolution, size, log-odds, and thresholds 
        '''
        self.res = 20 # cells per meter
        self.size = 60 # meters
        self.grid = np.zeros((self.res*self.size, self.res*self.size)) # log odds
        self.occ = 0.874
        self.free = -0.426
        self.occ_thres = 2.197
        self.free_thres = -1.386
        self.bound = 100 


class Particles(object):
    def __init__(self,nums,n_thres):
        '''
        Initialize Particles
        Output:
            Particles - Dictionary containing number of particles, weights of each, state of each, noise
        '''
        self.nums = nums
        self.weights = np.ones(self.nums) / self.nums
        self.states = np.zeros((3, self.nums))
        self.n_cov = [0.005, 0.005, 0.005] # [0.005, 0.005, 0.005]
        self.n_thres = n_thres