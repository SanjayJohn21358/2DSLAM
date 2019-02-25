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

results/example20.png: final no texture image of dataset 20

results/example21.png: final no texture image of dataset 21

results/example23.png: final no texture image of dataset 23

results/video20.png: video with no texture of dataset 20

results/video21.png: video with no texture of dataset 21

results/video23.png: video with no texture of dataset 23

results/texture_example20.png: final texture image of dataset 20

results/texture_example21.png: final texture image of dataset 21

results/texture_video20.avi: video with texture of dataset 20

results/texture_video21.avi: video with texture of dataset 21


Red is trajectory, blue is dead reckoning.

![alt text](results/example20.png?raw=True 'Occupancy map, white is free space')

![alt text](results/texture_example20.png?raw=True 'Texture map')

