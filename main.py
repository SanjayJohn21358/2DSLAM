import numpy as np
import load_data as ld
from map_utils import *
import map_particle_class
import os


def run_slam(dataset,texture=True):
    ''' 
    Run Particle filter algorithm, and plot results
    '''
    # load and process data
    enc_data, imu_data, lidar_data, kinect_data = load_data(dataset,texture)
    
    # define properties of Particles
    num_particles = 100
    n_thres = 99.0

    # initialize Map and Particle objects 
    Map = map_particle_class.Map()
    DeadReckoning_Particle = map_particle_class.Particles(1,n_thres)
    Particles = map_particle_class.Particles(num_particles,n_thres)

    # keeps track of particle position
    dead_reckoning = []
    trajectory = []

    # make plot
    h, w = Map.grid.shape
    plot = np.zeros((h,w,3),np.uint8)

    #run through all scans
    num_scans = lidar_data['ranges'].shape[1]
    speed = 1   # number of scans to skip, 1 = 0
    for lidar_idx in range(0, num_scans, speed):

        # pick best particle for next round (part of update step)
        state = Particles.states[:, np.argmax(Particles.weights)]

        # add state of particle to trajectory
        dead_reckoning.append(np.copy(DeadReckoning_Particle.states[:,0]))
        trajectory.append(np.copy(state))

        # -------- Synchronization -------- #

        # extract lidar scan within range and transform to lidar's cartesian coordinate
        lidar_scan = lidar_data['ranges'][:,lidar_idx] #take each scan
        valid = np.logical_and(lidar_scan>=lidar_data['min'], lidar_scan<=lidar_data['max']) # remove invalid lidar scans
        lidar_hit = polar_to_cart(lidar_scan[valid], lidar_data['angles'][valid]) #change to x,y coordinates

        # find closest imu, enc to lidar
        imu_idx = np.argmin(np.abs(imu_data['ts']-lidar_data['ts'][lidar_idx]))
        imu_av = imu_data['av'][imu_idx]
        enc_idx = np.argmin(np.abs(enc_data['ts']-lidar_data['ts'][lidar_idx]))
        enc_ld_r = enc_data['ld_r'][enc_idx]
        enc_ld_l = enc_data['ld_l'][enc_idx]

        if texture == True:
            # find closest kinect data to lidar
            kinect_d_idx = np.argmin(np.abs(kinect_data['d_ts']-lidar_data['ts'][lidar_idx]))
            kinect_rgb_idx = np.argmin(np.abs(kinect_data['rgb_ts']-lidar_data['ts'][lidar_idx]))
            # find corresponding kinect image file
            kinect_d_file = 'dataRGBD/Disparity'+str(dataset)+'/disparity'+str(dataset)+'_'+str(kinect_d_idx+1)+'.png'
            kinect_rgb_file = 'dataRGBD/RGB'+str(dataset)+'/rgb'+str(dataset)+'_'+str(kinect_rgb_idx+1)+'.png'
        else:
            kinect_d_file = ''
            kinect_rgb_file = ''

        # --------  Mapping --------- #
        
        # transform hit from lidar to world coordinate
        world_hit = lidar_to_world(lidar_hit, state=state)

        # update map according to hit
        update_map(world_hit[:2], state[:2], Map)

        # if first scan return back to start
        if lidar_idx == 0:
            continue
        
        # -------- Prediction -------- #

        # use differential drive model (and IMU) to predict new position

        # dead reckoning (no update)
        DeadReckoning_Particle.states = diff_model_predict(DeadReckoning_Particle.states,imu_av,enc_ld_r,enc_ld_l,lidar_data['ts'][lidar_idx] - lidar_data['ts'][lidar_idx-1]).reshape(3,1)
        
        # find new state of particle to update particle 
        new_state = diff_model_predict(state,imu_av,enc_ld_r,enc_ld_l,lidar_data['ts'][lidar_idx] - lidar_data['ts'][lidar_idx-1])
        
        # -------- Update -------- #

        # update particle weights (find lidar hits for all particles, use lidar_to_world)
        particle_update(Particles, Map, lidar_hit, new_state)
        
        # ------- Plotting and Texture Mapping -------- #

        # write image
        write_image(dead_reckoning, Map, plot, trajectory, lidar_idx, kinect_rgb_file, kinect_d_file, new_state, dataset, texture)

        if lidar_idx%1000 == 0:
            print('Mapping scans: ' + str(lidar_idx) + '/' + str(num_scans))


def load_data(dataset,texture):
    '''
    Load and preprocess data
    Input:
        dataset - number of dataset to use
        texture - boolean to switch on texture mapping (include kinect data)
    Output:
        enc_data - Encoder data in dictionary, with left and right displacement, and timestamps
        imu_data -  IMU data in dictionary, with angular velocity (yaw rate), linear acceleration and timestamps
        lidar_data - LiDAR data in dictionary, with ranges, angles and timestamps
    '''
    #load data
    imu_data = ld.get_imu(dataset)
    lidar_data = ld.get_lidar(dataset)
    enc_data = ld.get_enc(dataset)
    if texture == True:
        kinect_data = ld.get_kinect_time(dataset)
    else:
        kinect_data = 0

    # remove bias for odometry, init state is (0,0,0)
    yaw_bias_av = np.mean(imu_data['av'][0:380])
    imu_data['av'] -= yaw_bias_av

    #apply band pass filter
    order = 1
    fs = 50  # sample rate, Hz
    low = 0.000001 # desired band frequency of the filter, Hz
    high = 10
    imu_data['av'] = butter_bandpass_filter(imu_data['av'],low,high,fs,order)

    return enc_data, imu_data, lidar_data, kinect_data


if __name__ == '__main__':
    dataset = 20
    run_slam(dataset,texture=True)
    pathIn= 'images' + str(dataset) +'/'
    pathOut = 'video' + str(dataset) + '.avi'
    fps = 50.0
    convert_frames_to_video(pathIn, pathOut, fps)