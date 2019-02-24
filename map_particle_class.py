import numpy as np
class Map(object):
    def __init__(self):
        '''
        Initialize Map
        Input:
            None
        Output:
            Map - Map object with resolution, size, log-odds, and thresholds 
        '''
        self.size = 60 # meters
        self.res = 20 # cells per meter
        self.grid = np.zeros((self.res*self.size, self.res*self.size)) # occupancy grid
        prob_hit = 0.7 # prob of grid being occupied if lidar hit
        self.occ = np.log(prob_hit/(1-prob_hit)) # log odds of occupied value
        self.free = np.log((1-prob_hit)/(prob_hit)) # log odds of free value
        occ_thres = 0.9 # threshold value for occupied cells
        free_thres = 0.2 # threshold value for free cells
        self.occ_thres = np.log(occ_thres / (1 - occ_thres)) # log odds of threshold value
        self.free_thres = np.log(free_thres/(1-free_thres)) # log odds of threshold value
        self.bound = 100 #stops grid values from exploding


class Particles(object):
    def __init__(self,nums,n_thres):
        '''
        Initialize Particles
        Input:
            nums - number of particles
            n_thres - threshold for resampling
        Output:
            Particles - Particle object with weights, states
        '''
        self.nums = nums # number of particles
        self.weights = np.ones(self.nums) / self.nums # particle weights (starting at uniform)
        self.states = np.zeros((3, self.nums)) # particle state containing x, y, and theta
        self.n_thres = n_thres # threshold for 