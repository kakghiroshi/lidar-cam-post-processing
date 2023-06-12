
# coding:utf-8
import os
import sys
from tqdm import tqdm
from pinhole_process import pinhole_process
from fisheye_process import fisheye_process
import time 
import cv2
import yaml
import numpy as np
import threading
from multiprocessing import Pool

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
    #default
    #input folder include pcd and img

    input_path = "./data/"
    output_path = "./result/"

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    else:
        print("usage: python main.py <input_dir> <output_path>")
        print("-------------------------------------------------------------------------")
        print("default is ./data/ ./result/ \n")

    pinhole_pcd_dir = input_path + "/pinhole_pcd/"
    fisheye_pcd_dir = input_path + "/fisheye_pcd/"
    pinhole_img_left = input_path + "/pinhole/left/"
    pinhole_img_right = input_path + "/pinhole/right/"
    fisheye_img_left = input_path + "/fisheye/left/"
    fisheye_img_right = input_path + "/fisheye/right/"
    pinhole_LUT = "./rectification_file/pinhole/rectification.yml"
    fisheye_LUT = "./rectification_file/fisheye/rectification.yml"

    f1 = os.listdir(pinhole_pcd_dir)
    f1.sort()
    f2 = os.listdir(fisheye_pcd_dir)
    f2.sort()
    f3 = os.listdir(pinhole_img_right)
    f3.sort()
    f4 = os.listdir(fisheye_img_right)
    f4.sort()

    if (len(f1)!=len(f3)):
        print("The number of images on the left and right of the pinhole is not equal! ")
        sys.exit(0)
    
    if (len(f2)!=len(f4)):
        print("The number of images on the left and right of the fisheye is not equal! ")
        sys.exit(0)
    
    #pinhole output path
    pinhole_result_path = output_path + "/pinhole/"
    pinhole_depth_path = pinhole_result_path + "/depth/"
    pinhole_disparity_path = pinhole_result_path + "/disparity/"
    pinhole_projection_path = pinhole_result_path + "/projection/"
    rect_pinhole_left_path = pinhole_result_path + "/left/"
    rect_pinhole_right_path = pinhole_result_path + "/right/"
    
    #fisheye output path 
    fisheye_result_path = output_path + "/fisheye/"
    fisheye_depth_path = fisheye_result_path + "/depth/"
    fisheye_disparity_path = fisheye_result_path + "/disparity/"
    fisheye_projection_path = fisheye_result_path + "/projection/"
    rect_fisheye_left_path = fisheye_result_path + "/left/"
    rect_fisheye_right_path = fisheye_result_path + "/right/"
    
    fs = cv2.FileStorage(pinhole_LUT, cv2.FILE_STORAGE_READ)
    p_rect_x1 = fs.getNode("mapx1").mat()
    p_rect_y1 = fs.getNode("mapy1").mat()
    p_rect_x2 = fs.getNode("mapx2").mat()
    p_rect_y2 = fs.getNode("mapy2").mat()
    fs.release()

    fs1 = cv2.FileStorage(fisheye_LUT, cv2.FILE_STORAGE_READ)
    f_rect_x1 = fs1.getNode("mapx1").mat()
    f_rect_y1 = fs1.getNode("mapy1").mat()
    f_rect_x2 = fs1.getNode("mapx2").mat()
    f_rect_y2 = fs1.getNode("mapy2").mat()
    fs1.release()

    with open("./config/pinhole_config.yaml", "r") as stream:
            try:
                data = yaml.safe_load(stream)
                step_1 = data["step_1"]
                step_2 = data["step_2"]
                batch_1 = data["batch_1"]
                batch_2 = data["batch_2"]
                th1 = data["threshold_1"]
                th2 = data["threshold_2"]
            except yaml.YAMLError as exc:
                print(exc)

    with open("./config/fisheye_config.yaml", "r") as stream1:
            try:
                data1 = yaml.safe_load(stream1)
                step_3 = data1["step_1"]
                step_4 = data1["step_2"]
                batch_3 = data1["batch_1"]
                batch_4 = data1["batch_2"]
                th3 = data1["threshold_1"]
                th4 = data1["threshold_2"] 
            except yaml.YAMLError as exc:
                print(exc)

    print("pinhole data is processing ..")
    for i in tqdm(range(len(f1))):
        pinhole = pinhole_process(pinhole_pcd_dir, pinhole_img_left, pinhole_img_right, pinhole_LUT, pinhole_depth_path, pinhole_disparity_path, 
                                pinhole_projection_path, rect_pinhole_left_path, rect_pinhole_right_path,f1[i][0:-4])
        pinhole.create_dir()
        pinhole.load_yaml_file()
        pinhole.get_R()
        pinhole.project_points()
        pinhole.remap(p_rect_x1, p_rect_y1, p_rect_x2, p_rect_y2)
        
    print("fisheye data is processing ..")
    for j in tqdm(range(len(f2))):
        fisheye = fisheye_process(fisheye_pcd_dir, fisheye_img_left, fisheye_img_right, fisheye_LUT, fisheye_depth_path, fisheye_disparity_path, 
                                fisheye_projection_path, rect_fisheye_left_path, rect_fisheye_right_path, f2[j][0:-4])
        fisheye.create_dir()
        fisheye.load_yaml_file()
        fisheye.get_R()
        fisheye.EUCM_projection()
        fisheye.remap(f_rect_x1, f_rect_y1, f_rect_x2, f_rect_y2)

    f5=os.listdir(pinhole_depth_path)
    f5.sort()    
    f6=os.listdir(fisheye_depth_path)
    f6.sort()

    #multi threading process
    pool = Pool(processes=8)
    pool.map(remove_occlusion, [(f, pinhole_depth_path,step_1,batch_1,th1) for f in f5])
    pool.map(remove_occlusion, [(f, pinhole_depth_path,step_2,batch_2,th2) for f in f5])
    pool.map(remove_occlusion, [(f, fisheye_depth_path,step_3,batch_3,th1) for f in f6])
    pool.map(remove_occlusion, [(f, fisheye_depth_path,step_4,batch_4,th2) for f in f6]) 
    pool.close()
    pool.join()

    for m in tqdm(range(len(f1))):
        pinhole = pinhole_process(pinhole_pcd_dir, pinhole_img_left, pinhole_img_right, pinhole_LUT, pinhole_depth_path, pinhole_disparity_path, 
                                pinhole_projection_path, rect_pinhole_left_path, rect_pinhole_right_path,f1[m][0:-4])    
        pinhole.load_yaml_file()
        pinhole.depth_to_disparity()
        pinhole.visualize()
     
    for n in tqdm(range(len(f2))):
        fisheye = fisheye_process(fisheye_pcd_dir, fisheye_img_left, fisheye_img_right, fisheye_LUT, fisheye_depth_path, fisheye_disparity_path, 
                                fisheye_projection_path, rect_fisheye_left_path, rect_fisheye_right_path, f2[n][0:-4])
        fisheye.load_yaml_file()
        fisheye.depth_to_disparity()
        fisheye.visualize()

