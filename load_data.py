import numpy as np

def get_enc(dataset):
	with np.load("data/Encoders%d.npz"%dataset) as data:
		encoder_counts = data["counts"] # 4 x n encoder counts
		encoder_stamps = data["time_stamps"] # encoder time stamps

	enc_data = {}
	enc_data['ld'] = (((encoder_counts[0] + encoder_counts[2])*0.0022*0.5) + ((encoder_counts[1] + encoder_counts[3])*0.0022*0.5))*0.5
	enc_data['ts'] = encoder_stamps
	return enc_data

def get_lidar(dataset):
	lidar_data = {}
	with np.load("data/Hokuyo%d.npz"%dataset) as data:
		lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
		lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
		lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
		lidar_range_min = data["range_min"] # minimum range value [m]
		lidar_range_max = data["range_max"] # maximum range value [m]
		lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
		lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
	angles = np.arange(lidar_angle_min,lidar_angle_max + lidar_angle_increment,lidar_angle_increment)
	angles = angles[:-1] #remove extra angle
	lidar_data['angles'] = angles
	lidar_data['ranges'] = lidar_ranges
	lidar_data['ts'] = lidar_stamps
	lidar_data['min'] = lidar_range_min
	lidar_data['max'] = lidar_range_max
	return lidar_data

def get_imu(dataset):   
	imu_data = {} 
	with np.load("data/Imu%d.npz"%dataset) as data:
		imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
		imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
		imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
	imu_data['av'] = imu_angular_velocity[2]
	imu_data['la'] = imu_linear_acceleration[2]
	imu_data['ts'] = imu_stamps
	return imu_data
  

def get_kinect_time(dataset):
	kinect_data = {}
	with np.load("data/Kinect%d.npz"%dataset) as data:
		disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
		rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
	kinect_data['d_ts'] = disp_stamps
	kinect_data['rgb_ts'] = rgb_stamps
	return kinect_data
