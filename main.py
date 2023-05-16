import cv2
import torch
from PIL import Image
import os
import sys
from zoe.zoedepth.utils.misc import save_raw_16bit, colorize
from time import time
import glob
import matplotlib.pyplot as plt
import numpy as np
import requests
import json

PhoneCameraIP = "http://192.168.1.52:8080/shot.jpg"
VideoResolution = (1024, 768)

print("Welcome to Neudepth!")
print("Loading models and data...")

repo = "isl-org/ZoeDepth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VideoTempPath = "./vid_temp"
SaveRaw16Bit = True

model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
zoe = model_zoe_nk.to(DEVICE)

print("Done!")

def delete_folder_contents(folder: str):
    files = glob.glob("%s/*"%folder)
    for f in files:
        os.unlink(f)

def estimate_depth(img_path: str, out_path: str, save: bool = False):
    img = Image.open(img_path).convert("RGB")
    start_time = time()
    depth_pil = zoe.infer_pil(img)
    elapsed = time() - start_time
    print("\t> Depth estimation - elapsed: %.3f s"%elapsed)
    if save:
        if SaveRaw16Bit:
            save_raw_16bit(depth_pil, out_path)
        else:
            Image.fromarray(colorize(depth_pil)).save(out_path)
    return depth_pil

def estimate_depth_raw(img):
    depth = zoe.infer_pil(img)
    return colorize(depth)

def video_to_frames(video_path: str, frame_path: str, frame_rate: float = 0.5) -> list:
    """Returns list of paths to image frames"""
    files_generated = []
    try:
        print("Converting video to depth:\n\t> Video File: %s\n\t> Frame Rate (s): %.3f"%(video_path, frame_rate))
        os.makedirs(frame_path, exist_ok=True)
        delete_folder_contents(frame_path)
        
        vid_cap = cv2.VideoCapture(video_path)
        def getFrame(sec):
            vid_cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            return vid_cap.read()
        
        success = True
        cur_time = 0.0
        count = 0
        while success:
            success, image = getFrame(cur_time)
            if not success: break
            filename = "%s/rgb_%d.png"%(frame_path, count)
            filename_depth = "%s/d_%d.png"%(frame_path, count)
            cv2.imwrite(filename, image)
            files_generated.append(filename)          
            depth = estimate_depth(filename, filename_depth)
            depth = depth.astype(np.uint16)
            plt.imshow(depth, cmap="gray", vmin=0, vmax=4096)
            plt.show()
            print("\t> generated depth + rgb for frame %d at time %.3f"%(count, cur_time))              
            cur_time += frame_rate
            count += 1
    except Exception as e:
        print("Exception in video_to_frames(): %s"%str(e))
        sys.exit(-2)
    return files_generated

def grab_frame_from_phone():
    resp = None
    try:
        resp = requests.get(PhoneCameraIP, timeout=0.2)
        imgNp = np.array(bytearray(resp.content), dtype=np.uint8)
        if resp is not None: resp.close()
        return True, cv2.imdecode(imgNp, -1)
    except:
        if resp is not None: resp.close()
        return False, np.zeros((VideoResolution[1],VideoResolution[0],3), dtype=np.uint8)
    
def compute_depth_from_camera():
    while True:
        cv2.imshow("Webcam", estimate_depth_raw(grab_frame_from_phone()))
        if (cv2.waitKey(1) == 27):
            break
    cv2.destroyAllWindows()
    sys.exit(0)
    
    
def calibrate_camera(out_calib_file: str = "./out/camera_intrinsic.json"):
    CHECKERBOARD = (6, 9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    threedpoints = []
    twodpoints = []
    
    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0] 
                        * CHECKERBOARD[1], 
                        3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                                0:CHECKERBOARD[1]].T.reshape(-1, 2)    
    image = None
    grayColor = None
    blurry_threshold = 85.0
    
    print("Starting camera calibration, press Esc to stop...")
    cv2.imshow("Camera Calibration", np.zeros((VideoResolution[1],VideoResolution[0],3), dtype=np.uint8))
    print("\t> Waiting for camera connection...")
    while True:
        if (cv2.waitKey(1) == 27):
            break
        res, image = grab_frame_from_phone()
        if res:
            grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            variance = cv2.Laplacian(grayColor, cv2.CV_64F).var()
            if variance < blurry_threshold:
                continue
            
            ret, corners = cv2.findChessboardCorners( grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            if ret == True:
                threedpoints.append(objectp3d)
                corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
                
                twodpoints.append(corners2)
                
                image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
                
            cv2.imshow("Camera Calibration", image)
    cv2.destroyAllWindows()
    
    print("\t> Done gathering data, computing calibration...")
    
    if len (twodpoints) < 10:
        print("Insufficient number of calibration images... at least 10 required, got %u"%len(twodpoints))
        return
    
    if image is not None and grayColor is not None:
        h, w = image.shape[:2]
        
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)
        
        # Displaying required output
        print(" Camera matrix:")
        print(matrix)
        
        print("\n Distortion coefficient:")
        print(distortion)
        
        os.makedirs(os.path.dirname(out_calib_file), exist_ok=True)
        old_file = "%s.old"%out_calib_file
        if os.path.exists(out_calib_file):
            if os.path.exists(old_file): os.unlink(old_file)
            os.rename(out_calib_file, old_file)
        with open(out_calib_file, 'w') as outfile:
            obj = json.dump(
                {
                    'width': w,
                    'height': h,
                    'intrinsic_matrix': [
                        matrix[0][0], matrix[1][0], matrix[2][0], 
                        matrix[0][1], matrix[1][1], matrix[2][1], 
                        matrix[0][2], matrix[1][2], matrix[2][2]
                    ]
                },
                outfile,
                indent=4
            )
            
        print("calibration written to %s"%out_calib_file)

if __name__ == "__main__":
    
    calibrate_camera()
    sys.exit(0)

    
    if len(sys.argv) < 2:
        print("Not enough arguments")
        sys.exit(-1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print("Input file not found: %s"%input_file)
        
    # Is video? Convert to frames
    if ".mp4" in input_file:
        video_to_frames(input_file, VideoTempPath)
    #estimate_depth(load_image(sys.argv[1]))
        