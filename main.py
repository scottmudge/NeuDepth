import cv2
import torch
from PIL import Image
import os
import sys
from zoe.zoedepth.utils.misc import save_raw_16bit, colorize
import time
import glob
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
import dearpygui.dearpygui as dpg
import shutil
from threading import Thread

DefaultConfigPath = "./config/config.json"
DefaultCameraIntrinsicsFileName = "camera_intrinsic.json"
DefaultCameraIntrinsicsFilePath = "./config/%s"%DefaultCameraIntrinsicsFileName

class Config:
    DefaultConfigVars = {
        'camera_ip': '192.168.1.54',
        'camera_port': '8080',
        'camera_img_path': '/shot.jpg'
    }
    
    def __init__(self, path: str = DefaultConfigPath) -> None:
        self.vars = dict()
        self._path = path
        self.load()
        
    def get(self, key: str):
        if key in self.vars:
            return self.vars[key]
        elif key in Config.DefaultConfigVars:
            return Config.DefaultConfigVars[key]
        else:
            raise KeyError("key not found in config: %s"%key)
    
    def set(self, key: str, value):
        self.vars[key] = value
        self.save()
        
    def load(self):
        if os.path.exists(self._path):
            with open(self._path, 'r') as file:
                self.vars = json.load(file)
    
    def save(self):
        with open(self._path, 'w') as file:
            json.dump(self.vars, file, indent=4)    
            
class UI:
    def __init__(self) -> None:
        self.capture_sequence = False
        self.capture_sequence_stopped = True
    
    def load_ui(self):
        dpg.create_context()
        dpg.create_viewport()
        dpg.setup_dearpygui()
        
        with dpg.window(label="NeuDepth - Main", no_resize=True, no_collapse=True, no_close=True):
            dpg.add_text("NeuDepth - Monocular 3D Reconstruction")
            dpg.add_text("Note: Use IP Webcam app on Android")
            dpg.add_input_text(label="Camera IP", default_value=config.get('camera_ip'), callback=self._ip_changed, decimal=True, no_spaces=True)
            dpg.add_input_text(label="Camera Port", default_value=config.get('camera_port'), callback=self._port_changed, decimal=True, no_spaces=True)
            dpg.add_button(label="Calibrate Camera", callback=self._calibrate_camera)
            dpg.add_button(label="Capture Scan Sequence", callback=lambda: dpg.configure_item('scan_capture_window', show=True, pos=[200, 200]))
            
        with dpg.window(label="Scan Sequence Capture", tag="scan_capture_window", no_resize=True, no_close=True, no_collapse=True, show=False):
            dpg.add_text("Press start to begin capture, be sure to start camera server")
            dpg.add_input_text(label="Captured Frames", tag="captured_frames", readonly=True, default_value="0")
            dpg.add_separator()
            dpg.add_button(label="Start Capture", tag='start_capture_button', callback=self._capture_button)
            
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
    
    def _ip_changed(self, sender, app_data):
        config.set('camera_ip', app_data)   
    
    def _port_changed(self, sender, app_data):
        config.set('camera_port', app_data)
    
    def _calibrate_camera(self):
        calibrate_camera()
        
    def _capture_button(self):
        if not self.capture_sequence:
            if not self.capture_sequence_stopped:
                return
            self.capture_sequence = True
            dpg.configure_item('start_capture_button', label='Stop Capture')
            thread = Thread(target=capture_image_sequence)
            thread.start()
        else:
            self.capture_sequence = False
            dpg.configure_item('start_capture_button', label='Start Capture')
        
ui = UI()

config = Config()
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
    start_time = time.time()
    depth_pil = zoe.infer_pil(img)
    elapsed = time.time() - start_time
    print("\t> Depth estimation - elapsed: %.3f s"%elapsed)
    if save:
        if SaveRaw16Bit:
            save_raw_16bit(depth_pil, out_path)
        else:
            Image.fromarray(colorize(depth_pil)).save(out_path)
    return depth_pil

def estimate_depth_raw(img, colorized: bool = False):
    depth = zoe.infer_pil(img)
    return colorize(depth) if colorized else depth

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
    addr = "http://%s:%s%s"%(config.get('camera_ip'), config.get('camera_port'), config.get('camera_img_path'))
    try:
        resp = requests.get(addr, timeout=0.2)
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

def capture_image_sequence():
    if not os.path.exists(DefaultCameraIntrinsicsFilePath):
        print("Cannot capture scan sequence! Please calibrate your camera first...")
        return
    
    ui.capture_sequence_stopped = False
    
    target_dir = "./scan_sequences/%d"%int(time.time())
    color_dir = "%s/color"%target_dir
    depth_dir = "%s/depth"%target_dir
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    shutil.copyfile(DefaultCameraIntrinsicsFilePath, "%s/%s"%(target_dir, DefaultCameraIntrinsicsFileName))    
    print("Starting scan sequence capture:\n\t> Dir: %s"%target_dir)
    
    cur_idx = 0
    
    while ui.capture_sequence:
        if (cv2.waitKey(1) == 27):
            ui.capture_sequence = False
            dpg.configure_item('start_capture_button', label='Start Capture')
            continue
        
        res, image = grab_frame_from_phone()
        if not res: continue
        depth = estimate_depth_raw(image)
        
        cv2.imshow("Color Image", image)
        cv2.imshow("Depth", colorize(depth))
        
        depth_filename = "%s/%s.png"%(depth_dir, str(cur_idx).zfill(6))
        rgb_filename = "%s/%s.jpg"%(color_dir, str(cur_idx).zfill(6))
        
        save_raw_16bit(depth, depth_filename)
        cv2.imwrite(rgb_filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                
        cur_idx += 1    
        
    print("Scan sequence capture ended...")
    cv2.destroyAllWindows()        
    ui.capture_sequence_stopped = True
        
    
def calibrate_camera(out_calib_file: str = DefaultCameraIntrinsicsFilePath):
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
    ui.load_ui()