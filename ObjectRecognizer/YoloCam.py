
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

subprocess.run('taskkill /f /fi "IMAGENAME eq CarlaUE4*"', shell=True)
os.startfile("""C:\carla\CarlaUE4_Low""")

while True:
    try:
        user_entry = input("Press ENTER to start: \n")
        client = carla.Client('localhost', 2000)
        client.set_timeout(50)
        world = client.load_world('Town02')
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
while True:
    
    # Retrieve and reshape the image
    world.tick()
    new_matrix, exists_object, img = img_label_gen.create_label_matrix()
    if exists_object:
        print(new_matrix)
        #np.save(r'C:\carla\CarAware_IC\np_images\image_numpy.npy', new_matrix)
        # cv2.imshow('ImageWindowName',img)

# # # Create a queue to store and retrieve the sensor data
# image_queue = queue.Queue()
# camera.listen(image_queue.put)


# # Get the world to camera matrix
# world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# # Get the attributes from the camera
# image_w = camera_bp.get_attribute("image_size_x").as_int()
# image_h = camera_bp.get_attribute("image_size_y").as_int()
# fov = camera_bp.get_attribute("fov").as_float()

# # Calculate the camera projection matrix to project from 3D -> 2D
# K = build_projection_matrix(image_w, image_h, fov)

# # #### GETTING THE BOUNDING BOXES
# # 

# bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
# bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))

# objects = world.get_environment_objects(carla.CityObjectLabel.TrafficLight)
# objects.extend(world.get_environment_objects(carla.CityObjectLabel.TrafficSigns))
# edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

# object_labels_dict = {"TrafficLight": 1, "TrafficSigns": 1/2}

# while True:
#     # Retrieve and reshape the image
#     world.tick()
#     image = image_queue.get()

#     img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
#     copy_image = np.zeros((image.height, image.width))

#     # Get the camera matrix 
#     world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
#     exists_object = False

#     for _object in objects:

#         # Filter for distance from ego vehicle
#         if _object.transform.location.distance(camera.get_transform().location) < 25:

#             # Calculate the dot product between the forward vector
#             # of the vehicle and the vector between the vehicle
#             # and the bounding box. We threshold this dot product
#             # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
            
            
#             forward_vec = camera.get_transform().get_forward_vector()
#             ray = _object.transform.location - camera.get_transform().location

#             if forward_vec.dot(ray) > 1:
#                 # Cycle through the vertices
#                 exists_object = True
#                 bb = _object.bounding_box
#                 verts = [v for v in bb.get_world_vertices(carla.Transform())]
#                 #print(verts[0], verts[1])
#                 p0 = get_image_point(verts[0], K, world_2_camera)
#                 x_max = int(p0[0])
#                 x_min = int(p0[0])
#                 y_max = int(p0[1])
#                 y_min = int(p0[1])

#                 for vert in verts[1:]:
#                     p = get_image_point(vert, K, world_2_camera)
#                     # Find the rightmost vertex
#                     if p[0] > x_max:
#                         x_max = int(p[0])
#                     # Find the leftmost vertex
#                     if p[0] < x_min:
#                         x_min = int(p[0])
#                     # Find the highest vertex
#                     if p[1] > y_max:
#                         y_max = int(p[1])
#                     # Find the lowest  vertex
#                     if p[1] < y_min:
#                         y_min = int(p[1])
                
#                 width = int(x_max) - int(x_min)
#                 height = int(y_max) - int(y_min)
#                 try:
#                     input_array = np.ones((width, height,4))*255#object_labels_dict[str(_object.type)]
#                     copy_image[int(y_min): int(y_max), int(x_min): int(x_max),:] = input_array*object_labels_dict[str(_object.type)]
#                     img[int(y_min): int(y_max), int(x_min): int(x_max)] = np.ones((width, height))*object_labels_dict[str(_object.type)]*2
#                 except: 
#                     pass
#                 cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 0), 2)
#                 cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 0), 2)
#                 cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 0), 2)
#                 cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 0), 2)
                
#                 # verts = [v for v in bb.get_world_vertices(carla.Transform())]
#                 # x_min = 10000
#                 # x_max = -10000
#                 # y_min = 10000
#                 # y_max = -10000
#                 # for edge in edges:
#                 #     # Join the vertices into edges
#                 #     p1 = get_image_point(verts[edge[0]], K, world_2_camera)
#                 #     p2 = get_image_point(verts[edge[1]],  K, world_2_camera)
#                 #     for p in [p1, p2]:
#                 #         if p[0] < x_min:
#                 #             x_min = int(p[0])
#                 #         if p[0] > x_max:
#                 #             x_max = int(p[0])
#                 #         if p[1] < y_min:
#                 #             y_min = int(p[1])
#                 #         if p[1] > y_max:
#                 #             y_max = int(p[1])
#                 #     # Draw the edges into the camera output
#                 #     try:
#                 #         copy_image[y_min: y_max, x_min: x_max,:] = img[y_min: y_max, x_min: x_max,:]
#                 #     except:
#                 #         pass
#                 #     cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (0,0,255, 255), 1)

#     # Now draw the image into the OpenCV display window
#     cv2.imshow('ImageWindowName',img)
#     copy_image = copy_image.astype(int)
#     if exists_object:
#         _time = datetime.now().strftime("%Y_%m_%d") 
#         np.savetxt(f"C:\carla\images\image_{_time}", copy_image, delimiter=',')
#     # Break the loop if the user presses the Q key
#     if cv2.waitKey(1) == ord('q'):
#         pass
#         break

# # Close the OpenCV display window when the game loop stops
# cv2.destroyAllWindows()






