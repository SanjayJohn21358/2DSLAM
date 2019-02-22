import numpy as np
import load_data as ld
from map_utils import *
from scipy.signal import butter, lfilter, freqz
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
    n_thres = 95

    # initialize Map and Particle objects 
    Map = map_particle_class.Map()
    Particles = map_particle_class.Particles(num_particles,n_thres)
    DeadReckoning_Particle = map_particle_class.Particles(1,n_thres)
    Trajectory = []
    DeadReckoning = []

    # init plot
    h, w = Map.grid.shape
    Plot = np.zeros((h,w,3),np.uint8)
    num_scans = lidar_data['ranges'].shape[1]


    #run through all scans
    speed = 1   # number of scans to skip, 1 = 0
    for lidar_idx in range(0, num_scans, speed):

        # pick best particle
        state = Particles.states[:, np.argmax(Particles.weights)]

        # add state of particle to trajectory
        DeadReckoning.append(np.copy(DeadReckoning_Particle.states[:,0]))
        Trajectory.append(np.copy(state))

        # --------  Mapping --------- #

        # extract lidar scan within range and transform to lidar's cartesian coordinate
        lidar_scan = lidar_data['ranges'][:,lidar_idx] #take each scan
        good_range = np.logical_and(lidar_scan>=lidar_data['min'], lidar_scan<=lidar_data['max']) # remove invalid lidar scans
        lidar_hit = polar_to_cart(lidar_scan[good_range], lidar_data['angles'][good_range]) #change to x,y coordinates

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
        
        # transform hit from lidar to world coordinate
        world_hit = lidar_to_world(lidar_hit, state=state)

        # update map according to hit
        update_map(world_hit[:2], state[:2], Map)

        # if first scan return back to start
        if lidar_idx == 0:
            continue
        
        # -------- Localization -------- #

        # use differential drive model (and IMU) to predict new position

        # dead reckoning (no update)
        DeadReckoning_Particle.states = diff_model(DeadReckoning_Particle.states,imu_av,enc_ld_r,enc_ld_l,lidar_data['ts'][lidar_idx] - lidar_data['ts'][lidar_idx-1]).reshape(3,1)
        
        # find new state of particle and update particle 
        new_state = diff_model(state,imu_av,enc_ld_r,enc_ld_l,lidar_data['ts'][lidar_idx] - lidar_data['ts'][lidar_idx-1])
        particle_update(Particles, Map, world_hit, new_state)
        
        # write image
        write_image(DeadReckoning, new_state, Map, Trajectory, world_hit, Plot, lidar_idx, kinect_rgb_file, kinect_d_file, dataset, texture)

        if lidar_idx%1000 == 0:
            print('Mapping scans: ' + str(lidar_idx) + '/' + str(num_scans))


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    Define and implement Butterworth lowpass filter
    Input:
        data - data to be filtered
        cutoff - cutoff frequency in Hz
        fs - sampling rate
        order - order of filter
    Output:
        y - output of filter
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low,high], btype='band', analog=False)
    y = lfilter(b, a, data)
    return y


def load_data(dataset,texture):
    '''
    Load and preprocess data
    Output:
        enc_data: Encoder data in dictionary, with left and right displacement, and timestamps
        imu_data: IMU data in dictionary, with angular velocity (yaw rate), linear acceleration and timestamps
        lidar_data: LiDAR data in dictionary, with ranges, angles and timestamps
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

    #apply low pass filter
    order = 1
    fs = 50      # sample rate, Hz
    low = 0.000001 # desired band frequency of the filter, Hz
    high = 10
    imu_data['av'] = butter_bandpass_filter(imu_data['av'],low,high,fs,order)

    return enc_data, imu_data, lidar_data, kinect_data


if __name__ == '__main__':
    dataset = 21
    run_slam(dataset,texture=False)
    pathIn= 'images' + str(dataset) +'/'
    pathOut = 'video' + str(dataset) + '.avi'
    fps = 50.0
    convert_frames_to_video(pathIn, pathOut, fps)