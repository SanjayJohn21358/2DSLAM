import numpy as np
from scipy.signal import butter, lfilter, freqz
import cv2
import os
from os.path import isfile, join
from PIL import Image


def bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
        (sx, sy)	start point of ray
        (ex, ey)	end point of ray
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
        dx,dy = dy,dx # swap 

    if dy == 0:
        q = np.zeros((dx+1,1))
    else:
        q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
    if steep:
        if sy <= ey:
            y = np.arange(sy,ey+1)
        else:
            y = np.arange(sy,ey-1,-1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx,ex+1)
        else:
            x = np.arange(sx,ex-1,-1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    return np.vstack((x,y))

def mapCorrelation(im, x_im, y_im, vp, xs, ys):
    '''
    INPUT 
    im              the map 
    x_im,y_im       physical x,y positions of the grid map cells
    vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
    xs,ys           physical x,y,positions you want to evaluate "correlation" 

    OUTPUT 
    cpr             sum of the cell values of all the positions hit by range sensor
    '''
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy] # 1 x 1076
        iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
        x1 = vp[0,:] + xs[jx] # 1 x 1076
        ix = np.int16(np.round((x1-xmin)/xresolution))
        valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
                                    np.logical_and((ix >=0), (ix < nx)))
        cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    '''
    Create Butterworth bandpass filter for filtering IMU data
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

def polar_to_cart(scan, angles):
    '''
    Converts polar coordinates to cartesian coordinates
    Input:
        scan - LiDAR range
        angles - LiDAR angles
    Output:
        array [x,y] - array of x,y coordinates corresponding to polar coords
    '''
    x = scan*np.cos(angles)
    y = scan*np.sin(angles)
    return np.vstack((x, y))

def lidar_to_world(lidar_hit, state=None, Particles=None):
    '''
    Converts LiDAR x,y coordinates from body frame to world frame
    Input:
        lidar_hit - Collection of x,y coordinates from lidar scan in body frame
        state - state of highest weight particle
        Particles - All particles in filter
    Output:
        world_hit - Collection of x,y coordinates from lidar scan in map frame
    '''

    # lidar to body
    y_bl = 0.136 #offset from lidar to middle of robot
    T_bl = np.array([[1,0,0],[0,1,y_bl],[0,0,1]]) 

    # when no particle is passed, this is used for mapping
    if Particles is None:
        # body to world
        x_wb = state[0]
        y_wb = state[1]
        yaw_wb = state[2]
        H_wb = np.array([[np.cos(yaw_wb), -np.sin(yaw_wb), x_wb],[np.sin(yaw_wb), np.cos(yaw_wb), y_wb],[0, 0, 1]])
  
        # lidar to body to world
        H_wl = H_wb.dot(T_bl)
        lidar_hit = np.vstack((lidar_hit,np.ones((1,lidar_hit.shape[1]))))
        world_hit = np.dot(H_wl, lidar_hit)

        # remove very close hits (like ground)
        valid = world_hit[2]>0.01
        world_hit = world_hit[:,valid]

        return world_hit[:3,:]

    # when particle is passed, this allows for correlation with map
    else: 
        nums = Particles.nums
        states = Particles.states
        particles_hit = []
        if lidar_hit.shape[0] < 3:
            lidar_hit = np.vstack((lidar_hit, np.ones((1, lidar_hit.shape[1]))))
        for i in range(nums):
            # body to world
            x_wb = states[0,i]
            y_wb = states[1,i]
            yaw_wb = states[2,i]
            H_wb = np.array([[np.cos(yaw_wb), -np.sin(yaw_wb), x_wb],[np.sin(yaw_wb), np.cos(yaw_wb), y_wb],[0, 0, 1]])

            # lidar to body to world
            H_wl = H_wb.dot(T_bl)
            world_hit = np.dot(H_wl, lidar_hit)[:3,:]

            # remove very close hits (like ground)
            valid = world_hit[2] > 0.01
            particles_hit.append(world_hit[:, valid])

        return np.transpose(np.asarray(particles_hit), (1,2,0))

    
        
def world_to_map(xy_w,Map):
    '''
    Convert world coordinates to pixels in map
    Input:
        xy_w - x,y-coordinates in world frame
        Map - Map dictionary containing log-odds grid
    Output:
        pixels - pixels to change in map
    '''
    # make pixel grid of zeros
    pixels = np.zeros(xy_w.shape, dtype=int)

    # coordinates to pixels 
    # make 0,0 correspond to center of grid
    # flip map around so that forward is left on map)
    pixels[0] = ((xy_w[0] + Map.size/2)/Map.res).astype(np.int)
    pixels[1] = ((-xy_w[1] + Map.size/2)/Map.res).astype(np.int)

    # make sure pixels stay inside grid to avoid indexing errors
    valid = np.logical_and.reduce([pixels[0] >= 0, pixels[0] < Map.size/Map.res, pixels[1] >= 0, pixels[1] < Map.size/Map.res])
    pixels = pixels[:,valid]

    return pixels

def update_map(hit, state, Map):
    '''
    Update Map grid containing log-odds with current hits and state
    Input:
        hit - world map coordinates of LiDAR scan hits
        state - state of highest weight particle
        Map - Map object containing log odds to be updated
    Output:
        None
    '''
    # transform world coordinates of lidar hits into cells that are occupied
    occupied_cells = world_to_map(hit, Map)

    # add log odds value for occupied grid
    Map.grid[occupied_cells[1], occupied_cells[0]] += Map.occ - Map.free #add back in draw contours

    # draw contour to map region between particle and lidar hits
    mask = np.zeros(Map.grid.shape)
    contour = np.hstack((world_to_map(state, Map).reshape(-1,1), occupied_cells))
    cv2.drawContours(image = mask, contours = [contour.T], contourIdx = -1, color = Map.free, thickness = -1)
    Map.grid += mask

    # stop grid values from exploding
    Map.grid[Map.grid>Map.bound] = Map.bound
    Map.grid[Map.grid<-Map.bound] = -Map.bound


def diff_model_predict(current_state,w,ld_r,ld_l,t):
    '''
    Predict new x,y and theta according to differential drive model (and IMU measurements)
    Input:
        current_state - x, y and theta of current state
        w - angular velocity (yaw rate) from IMU
        ld_r - linear displacement from right encoders
        ld_l - lienar displacement from left encoders
        t - time between measurements
    Output:
        new_state - x, y, theta of new state
    '''
    # get current state of particle
    x = current_state[0]
    y = current_state[1]
    theta = current_state[2]

    # convert encoder displacement to angular velocity
    ld = (ld_r + ld_l)/2.0
    enc_w = (ld_r/t - ld_l/t)/(0.311)
    
    # update under differential drive model
    new_x = x + ld*(np.sin(w/2))*np.cos(theta + w/2)/(w/2)
    new_y = y + ld*(np.sin(w/2))*np.sin(theta + w/2)/(w/2)

    # use both IMU and encoder, but weigh it towards IMU
    new_theta = theta + t*(0.98*w + 0.02*enc_w)

    return np.hstack((new_x,new_y,new_theta))


def particle_update(Particles, Map, lidar_hit, new_state):
    '''
    Update particles weight and position according to lidar hits and differential drive model
    Input:
        Particles - All particles in filter
        Map - Map dictionary containing logodds
        lidar_hit - LiDAR hits
        new_state - Predicted new state of particle 
    Output:
        None
    '''    
    
    # encoder noise
    noise = np.random.normal(0, 0.001)

    # update state
    Particles.states[0] = new_state[0] + 0.05*noise
    Particles.states[1] = new_state[1] + 0.05*noise
    Particles.states[2] = new_state[2] + 0.05*np.random.normal(0, 0.005)

    # get particle's lidar hits
    particles_lidar_hit = lidar_to_world(lidar_hit, Particles=Particles)

    # get correlation from map and particle lidar reading
    # mapCorrelation returned negative values, so did not use it
    corr = np.zeros(Particles.nums)
    for i in range(Particles.nums):
        occ = world_to_map(particles_lidar_hit[:2,:,i], Map)
        corr[i] = np.sum(Map.grid[occ[1],occ[0]]>Map.occ_thres)

    # update particle weights with correlation and softmax function
    log_weight = corr + np.log(Particles.weights)
    Particles.weights = np.exp(log_weight - np.max(log_weight + np.log(np.sum(np.exp(log_weight - np.max(log_weight))))))

    # resample using stratified resampling
    sample_state = np.zeros(Particles.states.shape)
    n_eff = 1/np.sum(Particles.weights**2)
    if n_eff <= Particles.n_thres:
        N = Particles.nums
        j = 0
        c = Particles.weights[0]
        for k in range(N):
            u = np.random.uniform(0,1.0/N)
            beta = u + k/N
            while beta > c:
                j += 1
                c += Particles.weights[j]
            sample_state[:,k] = Particles.states[:,j]
        Particles.states = sample_state
        Particles.weights = np.ones(N)/N

def texture_mapping(rgb, depth, state, Map, plot):
    '''
    Convert kinect rgbd data from camera frame to body frame to world frame to map
    Input:
        rgb - rgb image
        depth - disparity image
        state - best particle's state
        Map - map object for converting to map 
        plot - updating plot
    Output:
        None
    '''
    # read images
    img = cv2.imread(rgb)
    dimg = Image.open(depth)
    dimg = np.array(dimg.getdata(), np.uint16).reshape(dimg.size[1], dimg.size[0])

    # get dimensions
    width = img.shape[1]
    height = img.shape[0]
    width_d = dimg.shape[1]
    height_d = dimg.shape[0]

    # convert from disparity pixel to depth
    pixels = []  
    colors_and_depth = []
    dd = (-0.00304*dimg + 3.31)
    depth = 1.03/dd

    def uv_to_rgbij(u,v,dd):
        rgbi = np.rint((u*526.37 + dd*(-4.5*1750.46) + 19276.0)/585.051)
        rgbj = np.rint((v*526.37 + 16662.0)/585.051)
        return rgbi, rgbj

    # get u,v pixels
    u = np.arange(0, width_d, 1)
    v = np.arange(0, height_d, 1)
    uu, vv = np.meshgrid(u, v)

    # get corresponding rgbi, rgbj pixels
    rgbi, rgbj = uv_to_rgbij(uu,vv,dd)

    # put optical and corresponding rgb into vectors
    disparity_points = np.vstack([uu.ravel(), vv.ravel(), np.ones(width_d*height_d)])
    colors_and_depth = np.vstack([rgbi.ravel(), rgbj.ravel(), depth.ravel()])

    # convert from pixel to optical frame
    # inverse camera matrix
    Kd = np.array([[585.051,0,242.9414],[0,585.051,315.838],[0,0,1]])  # get depth camera matrix
    Kd_inv = np.matrix(np.linalg.inv(Kd))  # inverse of depth camera matrix
    disparity_points = Kd_inv*disparity_points

    # inverse projection (multiply by depth)
    disparity_points = np.multiply(disparity_points, colors_and_depth[2,:])
    disparity_points = np.vstack((disparity_points, np.ones((1,np.shape(disparity_points)[1]))))

    # convert optical to regular
    disparity_points = np.matrix([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])*disparity_points

    # convert camera to body frame 
    yaw_cb = 0.021
    pitch_cb = 0.36
    p_cb = [0.18,0.005,0.36]

    # define rotation matrices
    R_yaw = np.array([[np.cos(yaw_cb), -np.sin(yaw_cb), 0],[np.sin(yaw_cb), np.cos(yaw_cb), 0],[0, 0, 1]])    
    R_pitch = np.array([[np.cos(pitch_cb), 0, np.sin(pitch_cb)],[0,1,0],[-np.sin(pitch_cb), 0, np.cos(pitch_cb)]]) 
    R_cb = R_yaw*R_pitch
    H_cb = np.matrix(np.zeros((4,4)))
    H_cb[:3,:3] = R_cb
    H_cb[:3,3] = np.matrix(p_cb).T
    H_cb[3,3] = 1

    # convert body to world frame using current particle state
    x_wb = state[0] 
    y_wb = state[1]
    yaw_wb = state[2]
    H_wb = np.array([[np.cos(yaw_wb), -np.sin(yaw_wb), 0, x_wb],[np.sin(yaw_wb), np.cos(yaw_wb), 0, y_wb],[0, 0, 1, 0],[0,0,0,1]])    
    H_bw = np.linalg.inv(H_wb)

    # put it all together
    H_cw = H_cb*H_bw
    disparity_points = np.linalg.inv(H_cw)*disparity_points

    # threshold on Z-axis in world frame (height) to get ground plane
    ground_plane = np.logical_and((disparity_points[2] >= -1.5), (disparity_points[2] <= 1.5))

    disparity_points = np.vstack((disparity_points[0, :][ground_plane], disparity_points[1, :][ground_plane],
                             disparity_points[2, :][ground_plane], disparity_points[3, :][ground_plane]))
                            
    colors_and_depth = np.vstack((colors_and_depth[0].reshape(1,-1)[ground_plane], colors_and_depth[1].reshape(1,-1)[ground_plane],
                             colors_and_depth[2].reshape(1,-1)[ground_plane]))

    # X and Y in world coordinate corresponds to map coordinates
    pixels = world_to_map(disparity_points[0:2,:],Map)
    y = pixels[0,:].astype(int)
    x = pixels[1,:].astype(int)

    # find corresponding rgb values for corresponding ground plane points
    u = colors_and_depth[0, :].astype(int)
    v = colors_and_depth[1, :].astype(int)
    valid_index = np.logical_and((u < width), (v < height))
    u = u[valid_index]
    v = v[valid_index]
    y = y[valid_index]
    x = x[valid_index]
    plot[x,y] = img[v,u,:]



def write_image(dead_reckoning, Map, plot, trajectory, idx, rgb, depth, state, dataset, texture=True):
    '''
    Plot Map grid, dead reckoning, trajectory, texture and write image
    Input:
        dead_reckoning - dead reckoning particle (no update)
        Map - Map object
        plot - plot matrix to update
        trajectory - trajectory of particle
        idx - index of LiDAR to print progress
        rgb - rgb image to go into texture mapping
        depth - disparity image to go into texture mapping
        state - particle state for texture mapping
        dataset - number of dataset to write image number
        texture - boolean to switch on texture mapping
    Output:
        writes image
    '''
    # plot occupied pixels black
    occ_mask = Map.grid>Map.occ_thres
    plot[occ_mask] = [0,0,0] # black for occ

    # plot free pixels with texture map
    free_mask = Map.grid<Map.free_thres

    # plot either texture map or free pixels as white
    if texture == True:
        texture_mapping(rgb,depth,state,Map,plot)
    else:
        plot[free_mask] = [255,255,255] # white for free
    
    # plot the rest as gray
    rest_mask = np.logical_not(np.logical_or(occ_mask, free_mask))
    plot[rest_mask] = [128,128,128] # gray for unexplored
    
    # plot dead reckoning
    dead = np.asarray(dead_reckoning)[:,:2].reshape(-1,2)
    dead_pixel = world_to_map(dead.T,Map)
    plot[dead_pixel[1],dead_pixel[0]] = [255,0,0] # blue for dead reckoning

    # plot trajectory
    traj = np.asarray(trajectory)[:,:2]
    traj_pixel = world_to_map(traj.T, Map)
    plot[traj_pixel[1],traj_pixel[0]] = [0,0,255] # red for trajectory
    
    # write to file
    path = 'images' + str(dataset)
    cv2.imwrite(os.path.join(path ,'SLAM_'+str(idx)+'.png'), plot)

   
def convert_frames_to_video(pathIn,pathOut,fps):
    '''
    Function to convert frames into video
    Input:
        pathIn - input path for images
        pathOut - output path for video
        fps - frames per second of video
    Output:
        writes video
    '''
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f)) and join(pathIn,f)[-4:] == '.png']
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
    filename=pathIn + files[0]
    img = cv2.imread(filename)
    height,width,layers = img.shape
    size = (width,height)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each file
        img = cv2.imread(filename)
        #inserting the frames into an image array
        out.write(img)
        if i%1000 == 0:
            print('Converting image frames: ' + str(i))
    out.release()