
# coding:utf-8
import os
import cv2
import numpy as np
import open3d as o3d
import sys
np.set_printoptions(suppress=True)
import math
import copy
import yaml
import time
import threading
from multiprocessing import Pool
from tqdm import tqdm
class pinhole_process:
    def __init__(self, pcd_path, left_img_path, right_img_path, LUT_path, depth_path, disparity_path, result_path, rect_left, rect_right, file_name):
        self.pcd_path = pcd_path
        self.left_img_path = left_img_path
        self.right_img_path = right_img_path
        self.R_T = []
        self.depth_path = depth_path
        self.LUT_path = LUT_path
        self.disparity_path = disparity_path
        self.result_path = result_path
        self.rect_left = rect_left
        self.rect_right = rect_right
        self.file_name = file_name
        self.k1, self.k2, self.p1, self.p2, self.k3 = 0, 0, 0, 0, 0
        self.fx, self.fy, self.cx, self.cy, self.baseline = 0, 0, 0, 0, 0
        self.step_1, self.step_2, self.th1, self.th2 = 0, 0, 0, 0
        self.rvecs = []
        self.tvecs = []

    def create_dir(self):
        # if diretory doesn't exist,create it. 
        if (os.path.isdir(self.depth_path) == False):
            os.makedirs(self.depth_path, mode = 0o777)
        if (os.path.isdir(self.disparity_path) == False):
            os.makedirs(self.disparity_path, mode = 0o777)
        if (os.path.isdir(self.result_path) == False):
            os.makedirs(self.result_path, mode =0o777)
        if (os.path.isdir(self.rect_left) == False):
            os.makedirs(self.rect_left, mode = 0o777)
        if (os.path.isdir(self.rect_right) == False):
            os.makedirs(self.rect_right, mode = 0o777)

    def load_yaml_file(self):
        with open("./config/pinhole_config.yaml", "r") as stream:
            try:
                data = yaml.safe_load(stream)
                self.k1 = data["k1"]
                self.k2 = data["k2"]
                self.p1 = data["p1"]
                self.p2 = data["p2"]
                self.k3 = data["k3"]
                self.fx = data["fx"]
                self.fy = data["fy"]
                self.cx = data["cx"]
                self.cy = data["cy"]
                self.baseline = data["baseline"]
                self.rvecs = data["rvecs"]
                self.tvecs = data["tvecs"]
                
            except yaml.YAMLError as exc:
                print(exc)

    #Get a 4X4 extrinsic matrix from the rotation and translation vectors
    def get_R(self):
        rotat = np.array(self.rvecs)
        trans = np.array(self.tvecs)
        rotat = rotat.reshape(1, 3)
        trans = trans.reshape(3, 1)
        rot,_ = cv2.Rodrigues(rotat)
        trans = np.matrix(trans, dtype = np.float64)
        tmp = np.hstack((rot, trans))
        tmp1 = [[0, 0, 0, 1]]
        self.R_T = np.vstack((tmp, tmp1))
        
    def project_points(self):

        camera_matrix = np.array([[self.fx, 0, self.cx],
                                  [0, self.fy, self.cy],
                                  [0, 0, 1]], dtype = np.float64)

        #load the pcd and transform it to camera coordinate
        point_cloud = o3d.io.read_point_cloud(self.pcd_path + self.file_name + ".pcd")
        point_cloud = point_cloud.transform(self.R_T)
        pc_as_np = np.array(point_cloud.points)
        
        #filter the pcd,shorten the process time
        vertical = np.arctan2(abs(pc_as_np[:, 1]), pc_as_np[:, 2])# vertical degree fov
        horizon = np.arctan2(abs(pc_as_np[:, 0]), pc_as_np[:, 2])# horizon degree fov
        threshold_vertical = np.pi/4 # The smaller the threshold, the fewer points 
        threshold_horizon=np.pi*5/24 
        pc_as_np = pc_as_np[(vertical<=threshold_vertical)&(horizon<= threshold_horizon), :]
        
        #create numpy array
        depth = np.zeros((1392, 976), np.float32)
        
        #projection
        for o in range(len(pc_as_np)):
            camera_points = pc_as_np[o].reshape(3,1)
            #add distortion
            uvw = camera_matrix.dot(camera_points)
            u = uvw[0] /uvw[2]
            v = uvw[1] /uvw[2]
            x = (u-self.cx) / self.fx
            y = (v-self.cy) / self.fy
            r2 = pow(x,2) + pow(y,2)
            x1 = x * (1 + self.k1 * r2 + self.k2 * r2 ** 2 ) + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x ** 2)
            y1 = y * (1 + self.k1 * r2 + self.k2 * r2 ** 2 ) + 2 * self.p2 * x * y + self.p1 * (r2 + 2 * y ** 2)
            u_distorted = self.fx * x1 + self.cx
            v_distorted = self.fy * y1 + self.cy
    
            #remove outlier
            if u_distorted > 976 or v_distorted > 1392 or u_distorted < 0 or v_distorted < 0:
                continue
            depth[int(v_distorted)][int(u_distorted)] = pc_as_np[o][2]
        np.save(self.depth_path + self.file_name + ".npy", depth)


    def remap(self,rect_x1,rect_y1,rect_x2,rect_y2):
        
        #remap left image
        left_img = cv2.imread(self.left_img_path + self.file_name + ".png")
        left_img = cv2.rotate(left_img, cv2.ROTATE_90_COUNTERCLOCKWISE)#for pinhole image ,rotate 90 degree to fit the LUT 
        rect_left_img = cv2.remap(src = left_img, map1 = rect_x1, map2 = rect_y1, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(self.rect_left + self.file_name + ".png", rect_left_img)

        #remap right image
        right_img = cv2.imread(self.right_img_path + self.file_name + ".png")
        right_img = cv2.rotate(right_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rect_right_img = cv2.remap(src = right_img, map1 = rect_x2, map2 = rect_y2, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(self.rect_right + self.file_name + ".png", rect_right_img)

        #remap depth numpy
        depth_npy = np.load(self.depth_path + self.file_name + ".npy")
        rect_npy = cv2.remap(src = depth_npy, map1 = rect_x1, map2 = rect_y1, interpolation = cv2.INTER_NEAREST)
        np.save(self.depth_path + self.file_name + ".npy", rect_npy)
    
    def depth_to_disparity(self):
        #depth to disparity 
        np.set_printoptions(threshold=np.inf)
        depth = np.load(self.depth_path + self.file_name + ".npy")
        np.seterr(divide = 'ignore')
        disparity  = np.zeros([depth.shape[0], depth.shape[1]], dtype = np.float32)
        disparity=(self.fx*self.baseline) / depth
        np.save(self.disparity_path + self.file_name + ".npy", disparity)

    def visualize(self):
        vis_npy = np.load(self.depth_path + self.file_name + ".npy")
        vis_img = cv2.imread(self.rect_left + self.file_name + ".png")
        vis_npy[vis_npy>10]=0 #filter the point which depth > 10 m
        # Calculate the color value corresponding to the depth
        depth_max = vis_npy.max()
        depth_min = vis_npy.min()
        depth_range = depth_max - depth_min
        depth_normalized = (vis_npy - depth_min) / depth_range
        depth_normalized = np.clip(depth_normalized, 0, 1)
        colors = np.zeros((depth_normalized.shape[0], depth_normalized.shape[1], 3), dtype=np.uint8)
        colors[:, :, 2] = (depth_normalized * 255).astype(np.uint8)
        colors[:, :, 1] = ((1 - depth_normalized) * 255).astype(np.uint8)

        # Map color values to specific color intervals
        color_intervals = [(0, (0, 0, 255)), (0.2, (0, 255, 165)), (0.4, (255, 255, 0)), (0.6, (255, 165, 0)), (0.8, (255, 0, 165)), (1, (255, 255, 255))]
        for i in range(len(color_intervals) - 1):
            #mask = (depth_normalized >= color_intervals[i][0]) & (depth_normalized < color_intervals[i + 1][0])
            #colors[mask, :] = color_intervals[i][1]
            interval_start, color_start = color_intervals[i]
            interval_end, color_end = color_intervals[i + 1]
            mask = (depth_normalized >= interval_start) & (depth_normalized < interval_end)
            if mask.any():
                alpha = (depth_normalized[mask] - interval_start) / (interval_end - interval_start)
                colors[mask, :] = np.uint8((1 - alpha)[:, np.newaxis] * color_start + alpha[:, np.newaxis] * color_end)

        #draw dots
        mask = (vis_npy != 0) & (~np.isinf(vis_npy))
        vis_img[mask, :] = colors[mask, :]
        for v_m, v_n in zip(*mask.nonzero()):
            cv2.circle(vis_img, (v_n, v_m), 1, tuple(colors[v_m, v_n].astype(int)))
        cv2.imwrite(self.result_path + self.file_name + ".png" , vis_img)


def remove_occlusion(args):
        s_o=time.clock()
        f, pinhole_depth_path ,step,batch,th1= args
        occ = np.load(pinhole_depth_path + f)
        h = occ.shape[0]
        w = occ.shape[1]
        count_1=h/step
        count_2=w/step
        #step1 will be used to remove occlusions in structured scenes,The larger step_1 is, the more point clouds are removed 
        #zero_mask1 = np.zeros([self.step_1,self.step_1])
        for m_1 in range (count_1):
            for n_1 in range (count_2):
                x_start, y_start = m_1*step, n_1*step
                x_end, y_end = x_start + batch, y_start + batch
                zero_mask = (occ[x_start : x_end, y_start : y_end] == 0)
                diffs_masked = np.ma.masked_where(zero_mask, occ[x_start : x_end, y_start : y_end])
                #Set the pixels whose difference value is greater threshold_1 (in config file) in the rectangular area to 0
                #The smaller threshold_1 and threshold_2 is,the more point clouds are removed,It indicates the threshold for removal,the unit is meter.
                if  np.ptp(diffs_masked)>th1:
                    diffs = diffs_masked - np.mean(diffs_masked)
                    mean_diff = np.mean(diffs)
                    threshold = 0.2* mean_diff  # The threshold will be 0.2 times the mean difference
                    mask = np.logical_and(diffs > threshold, diffs > 0)
                    mask[zero_mask] = False
                    occ[x_start : x_end, y_start : y_end][mask] = 0
        np.save(pinhole_depth_path + f ,occ)


if __name__ == "__main__":
    
    input_path = "./data/"
    output_path = "./result/"

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    else:
        print("usage: python main.py <input_dir> <output_path>")
        print("-------------------------------------------------------------------------")
        print("default is ./data/ ./result/")

    pinhole_pcd_dir = input_path + "/pinhole_pcd/"
    pinhole_img_left = input_path + "/pinhole/left/"
    pinhole_img_right = input_path + "/pinhole/right/"
    pinhole_LUT = "./rectification_file/pinhole/rectification.yml"

    #pinhole output path
    pinhole_result_path = output_path + "/pinhole/"
    pinhole_depth_path = pinhole_result_path + "/depth/"
    pinhole_disparity_path = pinhole_result_path + "/disparity/"
    pinhole_projection_path = pinhole_result_path + "/projection/"
    rect_pinhole_left_path = pinhole_result_path + "/left/"
    rect_pinhole_right_path = pinhole_result_path + "/right/"

    f1 = os.listdir(pinhole_pcd_dir)
    f1.sort()

    #read the Look up table
    fs = cv2.FileStorage(pinhole_LUT, cv2.FILE_STORAGE_READ)
    rect_x1 = fs.getNode("mapx1").mat()
    rect_y1 = fs.getNode("mapy1").mat()
    rect_x2 = fs.getNode("mapx2").mat()
    rect_y2 = fs.getNode("mapy2").mat()
    fs.release()
    
    with open("./config/pinhole_config.yaml", "r") as stream1:
            try:
                data1 = yaml.safe_load(stream1)
            
                step_1 = data1["step_1"]
                step_2 = data1["step_2"]
                batch_1 = data1["batch_1"]
                batch_2 = data1["batch_2"]
                th1 = data1["threshold_1"]
                th2 = data1["threshold_2"]
                
            except yaml.YAMLError as exc:
                print(exc)
    
    for i in tqdm(range(len(f1))):
        pinhole=pinhole_process(pinhole_pcd_dir, pinhole_img_left, pinhole_img_right, pinhole_LUT, pinhole_depth_path, pinhole_disparity_path, 
                                pinhole_projection_path, rect_pinhole_left_path, rect_pinhole_right_path, f1[i][0:-4])
        pinhole.create_dir()
        pinhole.load_yaml_file()
        pinhole.get_R()
        pinhole.project_points()
        pinhole.remap(rect_x1,rect_y1,rect_x2,rect_y2)
    
    f2=os.listdir(pinhole_depth_path)
    f2.sort()
    #multi threading process
    pool = Pool(processes=8)
    #pool.map(remove_occlusion, [(f, pinhole_depth_path,step_1,batch_1,th1) for f in f2])
    #pool.map(remove_occlusion, [(f, pinhole_depth_path,step_2,batch_2,th2) for f in f2])
    pool.close()
    pool.join()

    for q in tqdm(range(len(f1))):
        pinhole=pinhole_process(pinhole_pcd_dir, pinhole_img_left, pinhole_img_right, pinhole_LUT, pinhole_depth_path, pinhole_disparity_path, 
                                pinhole_projection_path, rect_pinhole_left_path, rect_pinhole_right_path, f1[q][0:-4])
        pinhole.load_yaml_file()
        pinhole.depth_to_disparity()
        pinhole.visualize()

