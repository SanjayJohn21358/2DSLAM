# 2D SLAM

## Project Description
Program that takes in LiDAR, IMU, encoder, and Kinect data and uses SLAM to create a video map of environment. We are given a robot's sensor data as it moves along an environment; our task is to create a 2D map of this environment.

## Methodology
Uses Particle Filter to make appropriate changes based on map

## Dataset 
LiDAR data is used for mapping. IMU and encoder data from wheels are used for localization.

## Files
map_particle_class.py: Class file to establish Map and Particle classes. Contains log odds parameters, grid parameters, resampling threshold, occupancy threshold, etc.

map_utils.py: Contains all helper functions used to employ SLAM. 

load_data.py: Data loader file, puts data into appropriate dictionaries.

main.py: Runs the SLAM algorithm. Run this file to see results!

## Results
Red is trajectory, blue is dead reckoning.

![alt text](results/example20.png?raw=True 'Occupancy map, white is free space')

![alt text](results/texture_example20.png?raw=True 'Texture map')

