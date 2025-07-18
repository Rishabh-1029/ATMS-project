# Libraries
import cv2
import math
import torch
import numpy as np
from  ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import csv

device = 0 if torch.cuda.is_available() else 'cpu'

# GPU availbility
print("GPU available : ",torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())



# Object Detection model yolov8n.pt + Deepsort Tracker
model_path = r"C:\Users\Rishabh Surana\Desktop\Projects\ATMS project\Yolo models\yolov8n.pt"
model = YOLO(model_path)
tracker = DeepSort(max_age=25)
allowed_obj_vehicle = [1,2,3,5,7] # 1 : Bicycle, 2 : Car, 3 : Bike/Motercycle, 5 : Bus, 7 : Truck.



# Feed Description 
# feed_ip_url = "rtsp://<username>:<password>@<ip_address>/enr/live/<cam_id>/<stream_id>" #Live stream from an IP camera using RTSP protocol.
feed_ip_url=r"C:\Users\Rishabh Surana\Desktop\Projects\ATMS project\test_data\test_video\cars.mp4"
src_pts = np.array([[245, 0], [330, 0], [740, 360], [0, 360]], np.float32)      # ROI coordinates
roi_polygon =src_pts.reshape((-1, 1, 2))
dst_width = 640                                                     # Frame width
dst_height = 360                                                    # Frame Height
speed_limit = 30                                                    # in KM/Hr
expected_movement = [0,-1]                                          # From Top to Bottom
meters_per_pixel = 50/360                                           # Streach -- px-length : Approx 764.0320673898px, physical-length : 125m 



# Video feed access
cap = cv2.VideoCapture(feed_ip_url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dst_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dst_height)
if not cap.isOpened():
    print("Error : Could not open video")
    exit()
else:
    print("Video is opened and working")
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None or np.isnan(fps):
    fps = 20
print("FPS : ",fps)



# Variables
car_coordinates = {}
roi_coords = []
latest_boxes_coords={}

stall_cars = [] 
opposite_cars = []
overspeed = {}
Heavy_vehicles = []
collision_record = {} 
vehicle_count ={}
counted_vehicle = set()
vehicle_data_record = []
logged_track_ids = {}
fieldnames = ["Transaction_ID", "Timestamp", "Track_ID", "Class", "Speed_kmph",
              "Is_Stalled", "Is_Opposite", "Is_Heavy", "Is_Overspeeding"]


# REPORT
csv_filename = f"{datetime.now().date()}_vehicle_report.csv"
folder_path = r"C:\Users\Rishabh Surana\Desktop\Projects\ATMS project\Vehicle Event Reports"
os.makedirs(folder_path, exist_ok=True)
full_path = os.path.join(folder_path, csv_filename)

def final_report(speed, track_id, class_name):
    is_overspeeding = speed > speed_limit
    is_stalled = track_id in stall_cars
    is_opposite = track_id in opposite_cars
    is_heavy = track_id in Heavy_vehicles

    event_occurred = is_overspeeding or is_heavy or is_opposite or is_stalled
    if not event_occurred:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    transaction_id = f"{track_id}_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}"
    
    new_data = {
        "Transaction_ID": transaction_id,
        "Timestamp": timestamp,
        "Track_ID": track_id,
        "Class": class_name,
        "Speed_kmph": round(speed, 2),
        "Is_Stalled": is_stalled,
        "Is_Opposite": is_opposite,
        "Is_Heavy": is_heavy,
        "Is_Overspeeding": is_overspeeding
    }

    if track_id not in logged_track_ids:
        logged_track_ids[track_id] = new_data
        write_to_csv(new_data)
    else:
        # Optional: update with new flags if required
        prev_data = logged_track_ids[track_id]
        updated = False
        if is_stalled and not prev_data["Is_Stalled"]:
            prev_data["Is_Stalled"] = True
            updated = True
        if is_opposite and not prev_data["Is_Opposite"]:
            prev_data["Is_Opposite"] = True
            updated = True
        if is_heavy and not prev_data["Is_Heavy"]:
            prev_data["Is_Heavy"] = True
            updated = True
        if is_overspeeding and not prev_data["Is_Overspeeding"]:
            prev_data["Is_Overspeeding"] = True
            updated = True
        if updated:
            prev_data["Speed_kmph"] = max(prev_data["Speed_kmph"], round(speed, 2))
            prev_data["Timestamp"] = timestamp
            write_to_csv(prev_data)

def write_to_csv(data):
    file_exist = os.path.exists(full_path)
    with open(full_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exist or os.path.getsize(full_path) == 0:
            writer.writeheader()
        writer.writerow(data)



# Event - Heavy Vehicle Detection
def heavy_vehicle(car_id):
    if(car_id not in Heavy_vehicles): 
        if(len(Heavy_vehicles)>30):
            Heavy_vehicles.pop(0)
        Heavy_vehicles.append(car_id)   



# Event - Vehicle Count
def vehicle_count_fn(track_id, class_name):
    if track_id not in counted_vehicle:
        counted_vehicle.add(track_id)
        if class_name not in vehicle_count:
            vehicle_count[class_name] = 1 
        else:
            vehicle_count[class_name] += 1



# Object Detection
def obj_Detection(results, detections):
    for box in results.boxes:
        x1, y1, x2, y2 = map(int,box.xyxy[0])
        class_id = int(box.cls[0])
        if class_id not in allowed_obj_vehicle:
          continue
        class_name = results.names[class_id]
        confidence = float(box.conf[0])
        detections.append(([x1,y1,x2-x1,y2-y1], confidence, class_name))



# Speed Calculation   
def calculate_speed(car_coordinates, fps, meters_per_pixel):
    speeds = []
    for i in range(1, len(car_coordinates)):
        x1, y1 = car_coordinates[i-1]
        x2, y2 = car_coordinates[i]
        pixel_dist = math.hypot(x2 - x1, y2 - y1)
        distance_m = pixel_dist * meters_per_pixel
        time_s = 1 / fps
        speed_mps = distance_m / time_s
        speed_kmph = speed_mps * 3.6
        speeds.append(speed_kmph)
    if speeds:
        return sum(speeds)/len(speeds)
    else:
        return 0.0



# Vehicle in ROI checking   
def is_in_polygon_roi(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0



# BEV Homography 
def get_bev_homography(src_pts, dst_width = dst_width, dst_height = dst_height):
  dst_pts = np.float32([ [0,0], [dst_width,0], [dst_width,dst_height],  [0,dst_height]])
  H, _ = cv2.findHomography(src_pts, dst_pts)
  return H, (dst_width, dst_height)

def apply_bev_transform(point, H):
  point = np.array([[point]], dtype='float32')
  transformed = cv2.perspectiveTransform(point, H)
  return transformed[0][0]
H, bev_size = get_bev_homography(src_pts)



# Event - Stall/Stop Vehicle detection - fps * 5 : If a car doesn't move for 5 seconds then it will be flaged Stalled.
def stall_vehicle_detection(x1,x2,y1,y2,car_id):
  def stall_movement_fn(car_coordinates, car_id):
      def eucledian(base_coord, pt):
          pt1 = (base_coord[0] - pt[0])**2
          pt2 = (base_coord[1] - pt[1])**2
          dist = math.sqrt(pt1+pt2)
          return dist

      coords = car_coordinates[car_id]
      stalled = True
      if(len(coords) > fps * 5):
          recent_coords = coords[-(fps * 5):]
          base_coord = recent_coords[0]
          for pt in recent_coords[1:]:
              distance = eucledian(base_coord, pt)
              if distance >= 1/meters_per_pixel: #Distance threshold 1 m
                  stalled = False
                  break
      else:
          stalled = False

      return stalled

  if(x2-x1 == 0 and y2-y1 == 0):
    if(stall_movement_fn(car_coordinates, car_id)):
      if(car_id not in stall_cars):
         if len(stall_cars)>50:
            stall_cars.pop(0)
         stall_cars.append(car_id)



# Event - Opposite Vehicle detection
def opposite_vehicles(x1,x2,y1,y2,expected,threshold,car_id):

    def normalize_vector(v):
          mag = math.sqrt(v[0]**2 + v[1]**2)
          if(mag!=0):
              return [v[0]/mag, v[1]/mag]
          else:
              return [0,0]
    def cosine_similarity(movment_vec, expected):
          dot_num = (movment_vec[0] * expected[0]) + (movment_vec[1] * expected[1])
          mag1 = math.sqrt((movment_vec[0]**2)+(movment_vec[1]**2))
          mag2 = math.sqrt((expected[0]**2)+(expected[1]**2))
          if(mag1 and mag2):
              return dot_num / (mag1 * mag2)
          else:
              return 1

    movment = [x2-x1, y2-y1]
    movement_unit = normalize_vector(movment)
    similarity = cosine_similarity(movement_unit, expected)

    if(similarity < -threshold):
        if(car_id not in opposite_cars):
            if(len(opposite_cars)>30):
                opposite_cars.pop(0)
            opposite_cars.append(car_id)


    
# Vehicle_Movement detection
def detect_car_movement(car_coordinates, expected = expected_movement, threshold = 0.75):
    for car_id, coords in car_coordinates.items():

        if(len(coords)<5):
            continue

        x1, y1 = coords[-5]
        x2, y2 = coords[-1]

        opposite_vehicles(x1,x2,y1,y2,expected,threshold,car_id)
        stall_vehicle_detection(x1,x2,y1,y2,car_id)    



# Car coordinates 
def car_coords_fn(track_id, x1, y1, x2, y2):
    center_x = int((x1 + x2)/2)
    center_y = int((y1 + y2)/2)
    bev_x, bev_y = apply_bev_transform((center_x, center_y), H)
    if track_id not in car_coordinates:
        car_coordinates[track_id] = []
    if len(car_coordinates[track_id]) >= 20:
        car_coordinates[track_id].pop(0)
    car_coordinates[track_id].append([bev_x, bev_y])
    detect_car_movement(car_coordinates)     


    
# Event - Accident Detection
def accident_detection():
  def cal_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if  interArea == 0:
     return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
  for id1 in stall_cars:
      for id2 in stall_cars:
          if(id1 == id2):
              continue
          if id1 not in latest_boxes_coords or id2 not in latest_boxes_coords:
              continue
          box1 = latest_boxes_coords[id1]
          box2 = latest_boxes_coords[id2]
          iou = cal_iou(box1, box2)
          if iou > 0.5:
            if id1 not in collision_record:
                    collision_record[id1] = []
            if id2 not in collision_record[id1]:
                collision_record[id1].append(id2)
  
  
        
# Event - Overspeed + Heavy Vehicle + Drawing Track
def draw_Tracks(tracks, frame, roi_polygon, counted_vehicles, fps):
    
    global logged_track_ids
    
    roi_polygon = np.array(roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [roi_polygon], isClosed=True, color=(255, 0, 0), thickness=1)

    vehicle_data_record = []

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        latest_boxes_coords[track_id] = [x1, y1, x2, y2]
        class_name = track.get_det_class()

        vehicle_count_fn(track_id, class_name)
        if class_name == "truck":
            heavy_vehicle(track_id)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if not is_in_polygon_roi(cx, cy, roi_polygon):
            continue

        car_coords_fn(track_id, x1, y1, x2, y2)
        speed = calculate_speed(car_coordinates[track_id], fps=fps, meters_per_pixel=meters_per_pixel)

        if speed > speed_limit and track_id not in overspeed:
            if len(overspeed) > 30:
                overspeed.pop(0)
            overspeed[track_id] = speed

        label = f"ID {track_id} | {class_name} | {speed:.1f} km/h"
        if track_id in stall_cars:
            color = (0, 255, 255)
        elif track_id in opposite_cars:
            color = (0, 0, 255)
        elif speed > speed_limit:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        final_report(speed, track_id, class_name) 
                
            
# MAIN_LOOP

print("\n---Feed Started---\n")
while True:
   ret, frame = cap.read()

   if not ret:
       cap.grab() #To skip frame
       continue
      
   frame = cv2.resize(frame, (dst_width, dst_height))
   results = model(frame, device=device, conf=0.6, verbose=False)[0]

   detections = []
   obj_Detection(results, detections)

   tracks = tracker.update_tracks(detections, frame = frame)
   draw_Tracks(tracks,frame,roi_polygon, counted_vehicle, fps)
   accident_detection()
   cv2.imshow("Live Detection", frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

total_vehicle = len(car_coordinates)
print(f"Total Vehicles = {total_vehicle}\n")
print("\n---Feed Ended---\n")
print("\n---Total Vehicle Count---\n")
for vehicle in vehicle_count:
    print(f"{vehicle} : {vehicle_count[vehicle]}")