
# coding:utf-8
import os
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time
import sys
np.set_printoptions(suppress=True)
import math
import yaml
from tqdm import tqdm
import threading
from multiprocessing import Pool

class fisheye_process:
    def __init__(self, pcd_path, left_img_path, right_img_path, LUT_path, depth_path, disparity_path, result_path, rect_left, rect_right, file_name):
        self.pcd_path = pcd_path
        self.left_img_path = left_img_path
        self.right_img_path = right_img_path
        self.LUT_path = LUT_path
        self.R_T = []
        self.depth_path = depth_path
        self.disparity_path = disparity_path
        self.result_path = result_path
        self.rect_left = rect_left
        self.rect_right = rect_right
        self.file_name = file_name
        self.k1, self.k2, self.k3, self.k4 = 0, 0, 0, 0
        self.fx, self.fy, self.cx, self.cy, = 0, 0, 0, 0
        self.step_1, self.step_2, self.th1, self.th2 = 0, 0, 0, 0
        self.rvecs = []
        self.tvecs = []
        self.rect = []
    
    def create_dir(self):
        # if diretory doesn't exist,create it. 
        if (os.path.isdir(self.depth_path) == False):
            os.makedirs(self.depth_path, mode = 0o777)
        if (os.path.isdir(self.disparity_path) == False):
            os.makedirs(self.disparity_path, mode = 0o777)
        if (os.path.isdir(self.result_path) == False):
            os.makedirs(self.result_path, mode = 0o777)
        if (os.path.isdir(self.rect_left) == False):
            os.makedirs(self.rect_left, mode = 0o777)
        if (os.path.isdir(self.rect_right) == False):
            os.makedirs(self.rect_right, mode = 0o777)

    def load_yaml_file(self):
        with open("./config/fisheye_config.yaml", "r") as stream:
            try:
                data = yaml.safe_load(stream)
                self.calib = data["calib"]
                self.rvecs = data["rvecs"]
                self.tvecs = data["tvecs"]
                self.rect = data["rect"]# rect image intrinsic
                self.step_1 = data["step_1"]
                self.step_2 = data["step_2"]
                self.th1 = data["threshold_1"]
                self.th2 = data["threshold_2"]
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
                
    def EUCM_projection(self):
        # read the image
        img = cv2.imread(self.left_img_path + self.file_name + ".png")
        point_cloud = o3d.io.read_point_cloud(self.pcd_path + self.file_name + ".pcd")
        point_cloud = point_cloud.transform(self.R_T)
        points = np.array(point_cloud.points) 
        depth = np.zeros((img.shape[0], img.shape[1]), np.float32)
        for p in points:
            r2 = p[0] * p[0] + p[1] * p[1]
            rho2 = self.calib[5] * r2 + p[2] * p[2]
            rho = math.sqrt(rho2)
            norm = self.calib[4] * rho + (1. - self.calib[4]) * p[2]
            mx = p[0] / norm
            my = p[1] / norm
            u=self.calib[0]*mx+self.calib[2]
            v=self.calib[1]*my+self.calib[3]
            if u > img.shape[1] or v > img.shape[0] or int(u)<0 or int(v)<0:
                continue
            distance = float(math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]))
            depth[int(v)][int(u)] = distance # use euclidean metric for disparity caculation
        np.save(self.depth_path + self.file_name + ".npy" , depth)
    
    def remap(self,rect_x1,rect_y1,rect_x2,rect_y2):
        
        #remap left image
        left_img = cv2.imread(self.left_img_path + self.file_name + ".png")
        rect_left_img = cv2.remap(src = left_img, map1 = rect_x1, map2 = rect_y1, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(self.rect_left + self.file_name + ".png", rect_left_img)

        #remap right image
        right_img = cv2.imread(self.right_img_path + self.file_name + ".png")
        rect_right_img = cv2.remap(src = right_img, map1 = rect_x2, map2 = rect_y2, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(self.rect_right + self.file_name + ".png", rect_left_img)

        #remap depth numpy
        depth_npy = np.load(self.depth_path + self.file_name + ".npy")
        rect_npy = cv2.remap(src = depth_npy, map1 = rect_x1, map2 = rect_y1, interpolation = cv2.INTER_NEAREST)
        np.save(self.depth_path + self.file_name + ".npy", rect_npy)

    def depth_to_disparity(self):
        #depth to disparity 
        depth = np.load(self.depth_path + self.file_name + ".npy")
        ori_input_h = depth.shape[0]
        ori_input_w = depth.shape[1]
        avg_input_h = ori_input_h/3
        disparity  = np.zeros([ori_input_h, ori_input_w], dtype = np.float32)
        part_dis = np.zeros([avg_input_h, ori_input_w], dtype = np.float32)
        np.seterr(divide = 'ignore')
        for k in range(3):
            k_start = k * avg_input_h
            k_end = k_start + avg_input_h
            partDepth = depth[k_start:k_end, :]
            #caculate the z value
            # X = x^2 = (u - cx)^2 / fx^2, Y = y^2 = (v - cy)^2 / fy^2, z=sqrt(distance^2-X-Y) 
            for j in range(avg_input_h):
                for i in range(ori_input_w):
                    X = math.pow((i - self.rect[2]), 2) / math.pow(self.rect[0], 2)#The X here actually refers to the square of the x value of the point in the world coordinate
                    Y = math.pow((j - self.rect[3]), 2) / math.pow(self.rect[1], 2)#As shown above, here represents the square of the y value
                    z = math.sqrt(math.pow(partDepth[j, i], 2) / (X + Y +1))
                    part_dis[j, i] = z
            disparity[k_start:k_end, :] = part_dis
        disparity = self.rect[4] * self.rect[0] / disparity            
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
    
    s_p=time.time()
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
    
    pcd_dir = input_path + "/fisheye_pcd/"
    fisheye_img_left = input_path + "/fisheye/left/"
    fisheye_img_right = input_path + "/fisheye/right/"
    fisheye_LUT = "./rectification_file/fisheye/rectification.yml"

    #fisheye output path 
    fisheye_result_path = output_path + "/fisheye/"
    fisheye_depth_path = fisheye_result_path + "/depth/"
    fisheye_disparity_path = fisheye_result_path + "/disparity/"
    fisheye_projection_path = fisheye_result_path + "/projection/"
    rect_fisheye_left_path = fisheye_result_path + "/left/"
    rect_fisheye_right_path = fisheye_result_path + "/right/"

    f1 = os.listdir(pcd_dir)
    f1.sort()

    fs = cv2.FileStorage(fisheye_LUT, cv2.FILE_STORAGE_READ)
    rect_x1 = fs.getNode("mapx1").mat()
    rect_y1 = fs.getNode("mapy1").mat()
    rect_x2 = fs.getNode("mapx2").mat()
    rect_y2 = fs.getNode("mapy2").mat()
    fs.release()

    with open("./config/fisheye_config.yaml", "r") as stream1:
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
        fisheye=fisheye_process(pcd_dir, fisheye_img_left, fisheye_img_right, fisheye_LUT, fisheye_depth_path, fisheye_disparity_path, 
                                fisheye_projection_path, rect_fisheye_left_path, rect_fisheye_right_path, f1[i][0:-4])
        fisheye.create_dir()
        fisheye.load_yaml_file()
        fisheye.get_R()
        fisheye.EUCM_projection()
        fisheye.remap(rect_x1,rect_y1,rect_x2,rect_y2)
    
    f2=os.listdir(fisheye_depth_path)
    f2.sort()
    #multi threading process
    pool = Pool(processes=8)
    pool.map(remove_occlusion, [(f, fisheye_depth_path,step_1,batch_1,th1) for f in f2])
    pool.map(remove_occlusion, [(f, fisheye_depth_path,step_2,batch_2,th2) for f in f2])
    pool.close()
    pool.join()

    for q in tqdm(range(len(f1))):
        fisheye=fisheye_process(pcd_dir, fisheye_img_left, fisheye_img_right, fisheye_LUT, fisheye_depth_path, fisheye_disparity_path, 
                                fisheye_projection_path, rect_fisheye_left_path, rect_fisheye_right_path, f1[q][0:-4])    
        fisheye.load_yaml_file()
        fisheye.depth_to_disparity()
        fisheye.visualize()
    e_p=time.time()
    print("run time is :",e_p-s_p)
