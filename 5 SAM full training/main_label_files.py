import pydicom as dicom
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import interpolate
from scipy.interpolate import *

import random




class DownsampleDataToFiles:
    def __init__(self, path_to_metadata, images_root, output_path, 
                 resize = (1024, 1024)):

        self.path_to_metadata = path_to_metadata
        self.images_root = images_root
        self.resize = resize
        self.output_path = output_path
        self.create_data_folder()

    # create common data directory
    def create_data_folder(self):
        os.makedirs(self.output_path, exist_ok=True)
  
    # read dicom, output resolution, numpy array and name
    def input_dicom(self, dicom_path):

        file = dicom.dcmread(os.path.join(self.images_root, dicom_path))
        try:
            self.res_x, self.res_y = file.ImagerPixelSpacing
        except:
            self.res_x, self.res_y = file.PixelSpacing
        self.image = file.pixel_array
        self.image_name = dicom_path.split(".dcm")[0]
        self.image_shape = self.image.shape
        print('image size', self.image.shape)
        #print('image resolutions', self.res_x, self.res_y)

    def dist_between_grid_points(self, np_point_one, np_point_two, res_x, res_y):
        # grid points should be 2D
        diff = np_point_one - np_point_two
        # print(diff)
        if len(diff) == 2:
            diff_scaled = diff * np.array([res_x, res_y])
        else: print('check dimensions of the input')
        return np.linalg.norm(diff_scaled)
    
    def apply_resize(self, image_array):
        return np.array(Image.fromarray(image_array).resize(self.resize, Image.Resampling.LANCZOS))
    
    def interpolate_one_curve(self, x, y, num, bc_type='natural'):
        points = np.array([x,y]).T
        # Linear length along the line:
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]

        # Cubic Interpolation 
        interpolator =  CubicSpline(distance, points, bc_type=bc_type)

        # 'num' point
        alpha = np.linspace(0, 1, num)
        interpolated_points = interpolator(alpha)
        interpolated_points = interpolated_points.T
        xyz = np.array([interpolated_points[0],interpolated_points[1]])

        return xyz.T
    
    def create_lm_masks(self, lm_locations, dist = 1.2): # here, dist is in mm
        # create masks for landmarks
        list_of_grids = []
        for lm in lm_locations:
            # print('landmark location', lm)
            x, y = lm
            x, y = int(x), int(y)
            grid = np.zeros(self.image.shape)
            if x >= self.image.shape[0] - 1:
                x =  self.image.shape[0] - 1
            if y >= self.image.shape[1] - 1: 
                y =  self.image.shape[1] - 1
            grid[x,y] = 1
            # for locations within 50 pixels from x
            for x_prime in range(max(0, x-50), min(grid.shape[0], x+50)):
                # for locations within 50 pixels from y
                for y_prime in range(max(0, y-50), min(grid.shape[1], y+50)):
                      # calculate distance to the lm
                    d = self.dist_between_grid_points(np.array([x_prime, y_prime]), lm, self.res_x, self.res_y)
                #   d = self.dist_between_grid_points(np.array([x_prime, y_prime]), lm, 
                #                                     self.res_x*self.image_shape[0]/self.resize[0], 
                #                                     self.res_y*self.image_shape[1]/self.resize[1])
                    # check the distance is less than
                    if d <= dist:
                        grid[int(x_prime),int(y_prime)] = 1
            list_of_grids.append(grid)   
        return list_of_grids
    
    def create_curve_mask(self, list_of_grids):
        label_array = np.sum(np.array(list_of_grids), axis = 0)
        return label_array
    

    def ind_work(self, ind):
        id = ind['id']
        
        # create individual folder
        id_path = os.path.join(self.output_path, id)
        os.makedirs(id_path, exist_ok=True)
        
        dicom_path = ind['image'][0]
        # transform image to numpy, read resolution
        self.input_dicom(dicom_path)
        res_x = float(self.res_x)
        res_y = float(self.res_y)
        
        # create folder for labels
        labels_path = os.path.join(id_path, 'labels')
        os.makedirs(labels_path, exist_ok=True)

        for key in ind['landmarks'].keys():
            if 'coordinates' in ind['landmarks'][key].keys():
                coords = ind['landmarks'][key]['coordinates']
                names = ind['landmarks'][key]['labels']
                
                if len(coords) > 0:
                    dims = len(coords[0])
                if dims == 3:
                    coords = [lm[:2] for lm in coords]

                # check the name of landmark collection
                if key not in ['LIST', 'Pelvis', 'PELVIS']:
                    # upsample outlines:
                    x_list = [lm[0] for lm in coords]
                    y_list = [lm[1] for lm in coords]

                    if key == 'Linea_terminalis':
                        coords = self.interpolate_one_curve(x_list, y_list, len(x_list)*40, bc_type='not-a-knot')
                    else: coords = self.interpolate_one_curve(x_list, y_list, len(x_list)*20)

                    if key == 'Calibration_Ball':
                        x_list.append(x_list[0])
                        print(x_list)
                        y_list.append(y_list[0])
                        coords = self.interpolate_one_curve(x_list, y_list, len(x_list)*20, bc_type='periodic')

                    # transform landmarks from mm to grid
                    lm_locations = [np.rint([x[1]/res_x, x[0]/res_y]) for x in  np.array(coords).astype(np.float32) ] # / np.array([res_x, res_y])
                    # create masks for landmarks
                    list_of_grids =  self.create_lm_masks(lm_locations)
                    # create one grid
                    curve_grid = self.create_curve_mask(list_of_grids)
                    # downsample the grid to resize
                    curve_grid = self.apply_resize(curve_grid)
                    # ensure only 0 and 1
                    curve_grid[curve_grid<0.5] = 0
                    curve_grid[curve_grid>=0.5] = 1
                    # save label as .npy file
                    np.save(os.path.join(labels_path, f'{key}.npy'), curve_grid)            

                else:  

                    # transform landmarks from mm to grid
                    lm_locations = [np.rint([x[1]/res_x, x[0]/res_y]) for x in  np.array(coords).astype(np.float32)] # / np.array([res_x, res_y])
                    # create masks for landmarks
                    list_of_grids =  self.create_lm_masks(lm_locations, dist=2.5)   

                    # resample landmark grids to resize
                    downsampled_lm_grids = [self.apply_resize(grid) for grid in list_of_grids]
                    for i, grid in enumerate(downsampled_lm_grids):
                        grid[grid<0.5] = 0
                        grid[grid>=0.5] = 1
                        np.save(os.path.join(labels_path, f'{key}_{names[i]}.npy'), grid)                   

            else: 
                continue

        # create image path and save image
        image_path = os.path.join(id_path, 'image')
        os.makedirs(image_path, exist_ok=True)

        downsampled_image = self.apply_resize(self.image)
        save_path_this_im = os.path.join(image_path, self.image_name + ".npz")
        np.savez(save_path_this_im, downsampled_image) # image will have to become RGB for SAM
        
        print(id, 'finished')

    # read metadata
    def apply(self):

        with open(self.path_to_metadata, "r") as j:
            metadata_all = json.loads(j.read())
            N = len(metadata_all['all'])
            print(N)

            for ind in metadata_all['all']:
                print('started', ind['id'])
                self.ind_work(ind)

            # any metadata?

# run the script
path_to_metadata = 'data/metadata_full.json'
images_root = 'data/root_image_folder'
output_path_anno = 'data/full'
create_labels_engine = DownsampleDataToFiles(path_to_metadata, images_root, output_path_anno)
create_labels_engine.apply()