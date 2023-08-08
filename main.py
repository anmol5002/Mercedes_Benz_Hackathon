from __future__ import division, print_function, absolute_import

NUSCENES_SDK_PATH = '/home/r3dg0li4th/DynamicSegmentationTool'
NUSCENES_DATABASE_PATH = '/home/r3dg0li4th/data/sets/nuscenes'
SCENE_NAME  = "scene-0061"

import datetime
from timeit import time
import warnings
import os
import sys
import statistics
import numpy as np
import cv2
from PIL import Image
# from dsort import obj_detect
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend
import os.path as osp
from yolo_detection import *
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ax = plt.axes(projection="3d")
sys.path.append(NUSCENES_SDK_PATH)

np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')
def main(yolo):
    start = time.time()
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.3
    counter = []
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    fps = 0.0
    nusc = NuScenes(dataroot=NUSCENES_DATABASE_PATH)
    nusc.list_scenes()
    scene_token = nusc.field2token('scene', 'name', SCENE_NAME)[0]
    scene = nusc.get('scene', scene_token)
    sample_token_itr = scene['first_sample_token']
    sample_count=0
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('./output/'+ "_" + "detect" + '_output.avi', fourcc, 15, (480, 360))
    frame_index = -1

    while sample_count<39:
        sample = nusc.get('sample', sample_token_itr)
        nusc.render_pointcloud_in_image(sample['token'],pointsensor_channel = 'RADAR_FRONT', camera_channel= 'CAM_FRONT', dot_size = 30)
        arr = np.genfromtxt('/home/r3dg0li4th/data/aa.csv', delimiter=',')
        cam = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        im = cv2.imread(osp.join(nusc.dataroot, cam['filename']))
        frame = im
        t11 = time.time()
        image = Image.fromarray(frame[...,::-1])
        boxs,class_names = yolo.detect_image(image)
        features = encoder(frame,boxs)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        tracker.predict()
        tracker.update(detections)
        i = int(0)
        indexIDs = []
        c = []
        boxes = []
        bb_data = []
        # print(bb_data)
        # print(arr)
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            temp_list = [[int(bbox[0]), int(bbox[3])], [int(bbox[2]), int(bbox[1])]]
            bb_data.append(temp_list)

        arr = arr.tolist()
        for t1 in arr:
            # print(t1[0]>=bb_data[0][0][0] and t1[0]<=bb_data[0][1][0] and t1[1]<=bb_data[0][0][1] and t1[1]>=bb_data[0][1][1])
            count=0
            for t2 in bb_data:
                if(t1[0]>=t2[0][0] and t1[0]<=t2[1][0] and t1[1]<=t2[0][1] and t1[1]>=t2[1][1]):
                    t2.append(t1[2:])
                bb_data[count] = t2
                count+=1

        # print(arr)
        count=0
        for t in bb_data:
            r_x=[]
            r_y=[]
            vx=[]
            vy=[]
            vx_comp=[]
            vy_comp=[]
            for t2 in t:
                if(len(t2)>2):
                    r_x.append(t2[0])
                    r_y.append(t2[1])
                    vx_comp.append(t2[2])
                    vy_comp.append(t2[3])
                    vx.append(t2[4])
                    vy.append(t2[5])
            if(len(r_x)!=0 and len(r_y)!=0 and len(vx)!=0 and len(vy)!=0 and len(vx_comp)!=0 and len(vy_comp)!=0):
                m_rx = gaussian_mean(r_x)
                m_ry = gaussian_mean(r_y)
                m_vx = gaussian_mean(vx)
                m_vy = gaussian_mean(vy)
                m_vx_comp = gaussian_mean(vx_comp)
                m_vy_comp = gaussian_mean(vy_comp)
                temp = [t[0],t[1],[round(m_rx,2),round(m_ry,2),round(m_vx,2),round(m_vy,2),round(m_vx_comp,2),round(m_vy_comp,2)]]
                bb_data[count]=temp
            count+=1

        for t in bb_data:
            if(len(t)>2):
                x=t[0][0]
                y0=t[1][1]
                x_plus_w=t[1][0]
                y_plus_h=t[0][1]
                label_top = "x:{},\ny:{},\nvx:{},\nvy:{}".format(t[2][0], t[2][1], t[2][2], t[2][3])
                dy = 20
                for j, line in enumerate(label_top.split('\n')):
                    y = y0 + j*dy
                    cv2.putText(im, line,(x,y-65),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,255,255),1,cv2.LINE_AA)

                label_bottom = "cvx:{},\ncvy:{}".format(t[2][4], t[2][5])
                for j, line in enumerate(label_bottom.split('\n')):
                    y = y_plus_h+35 + j*dy
                    cv2.putText(im, line,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7, (255,255,255),1,cv2.LINE_AA)


        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
            if len(class_names) > 0:
               class_name = class_names[0]
               cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)

            i += 1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            #center point
            cv2.circle(frame,  (center), 1, color, thickness)

        #draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
        count = len(set(counter))
        cv2.putText(frame, "Total Object Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "Current Object Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        cv2.namedWindow("YOLO3_Deep_SORT", 0);
        cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768);
        cv2.imshow('YOLO3_Deep_SORT', frame)

        out.write(frame)
        fps  = ( fps + (1./(time.time()-t11)) ) / 2

        sample_token_itr = sample['next']
        sample_count+=1
        # cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gaussian_mean(arr):
    gm=0
    mu = statistics.mean(arr)
    if(len(arr)!=1):
        sig = statistics.pstdev(arr)
        if(sig!=0):
            for t in arr:
                gm+=t*gaussian(t, mu, sig)
            return gm/len(arr)
        else:
            return mu
    else:
        return mu

if __name__ == '__main__':
    main(YOLO())
