
import carla
import math
import random
import time
import queue
import numpy as np
import cv2
import subprocess
import os
from camera_functions import get_image_point, build_projection_matrix
from datetime import datetime
from image_label_generator import ImageLabelGenerator
from PIL import Image
from collections import deque

subprocess.run('taskkill /f /fi "IMAGENAME eq CarlaUE4*"', shell=True)
os.startfile("""C:\carla\CarlaUE4_Low""")

while True:
    try:
        user_entry = input("Press ENTER to start: \n")
        client = carla.Client('localhost', 2000)
        client.set_timeout(50)
        world = client.load_world('Town01')
        break
    except Exception as e:
        user_entry = int(input("Write 0 to end the execution, 1 to try it again and 2 to keep running: \n"))
        if user_entry == 0:
            raise e
        elif user_entry == 2:
            break
        


bp_lib = world.get_blueprint_library()

# get possible spawn points
current_map = world.get_map()

# spawn_transforms will be a list of carla.Transform
spawn_points = current_map.get_spawn_points()

# spawn vehicle
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])


# # Create a queue to store and retrieve the sensor data
# image_queue = queue.Queue()
# camera.listen(image_queue.put)

# # spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
vehicle.set_autopilot(True)


# # Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
img_label_gen = ImageLabelGenerator(camera, world, 15, 5)
img_dequeue = deque([], maxlen=5)
camera.listen(img_dequeue.append)
while True:
    
    # Retrieve and reshape the image
    world.tick()
    if len(img_dequeue) > 0: 
        try:
            new_matrix, exists_object, img = img_label_gen.create_label_matrix(img_dequeue[-1])
            # img, new_matrix = img_label_gen.get_image_and_create_copy_zeros_matrix()
            if exists_object:
                # print(new_matrix)
                print(img_label_gen.create_labels_from_quadrants(img_label_gen.create_quadrants_from_matrix(new_matrix)))
                # Image.fromarray(new_matrix*127).show()
            cv2.imshow("Objects", new_matrix*127)
            cv2.imshow("Image", img)
            if cv2.waitKey(20) == ord('q'):
                break
        except:
            print("ERRO")
cv2.destroyAllWindows()







