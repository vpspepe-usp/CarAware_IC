from camera_functions import build_projection_matrix, get_image_point
import carla
import queue
import numpy as np
from typing import List

class ImageLabelGenerator:
    def __init__(self, camera, world, max_distance = 10, min_distance = 4.5):
        self.camera = camera 
        self.world = world
        # Get the attributes from the camera
        image_w = int(camera.attributes.get("image_size_x"))
        image_h = int(camera.attributes.get("image_size_y"))
        fov = float(camera.attributes.get("fov"))
        # Calculate the camera projection matrix to project from 3D -> 2D
        self.K = build_projection_matrix(image_w, image_h, fov)
        self.object_labels_dict = {"TrafficLight": 1, "TrafficSigns": 2}
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)
        self.max_distance = max_distance
        self.min_distance = min_distance
        
    def get_objects(self):
        objects = self.world.get_environment_objects(carla.CityObjectLabel.TrafficLight)
        objects.extend(self.world.get_environment_objects(carla.CityObjectLabel.TrafficSigns))
        return objects
    
    def get_image_and_create_copy_zeros_matrix(self):
        image = self.image_queue.get()
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        new_matrix = np.zeros((image.height, image.width))
        return img, new_matrix
    
    def get_x_y_min_max(self, world_2_camera, verts):
        p0 = get_image_point(verts[0], self.K, world_2_camera)
        x_max = int(p0[0])
        x_min = int(p0[0])
        y_max = int(p0[1])
        y_min = int(p0[1])
        for vert in verts[1:]:
            p = get_image_point(vert, self.K, world_2_camera)
            # Find the rightmost vertex
            if p[0] > x_max:
                x_max = int(p[0])
            # Find the leftmost vertex
            if p[0] < x_min:
                x_min = int(p[0])
            # Find the highest vertex
            if p[1] > y_max:
                y_max = int(p[1])
            # Find the lowest  vertex
            if p[1] < y_min:
                y_min = int(p[1])
        return x_max, x_min, y_max, y_min
    
    def add_object_label_into_new_matrix(
            self, _object, world_2_camera, new_matrix) -> np.ndarray:
        bb = _object.bounding_box
        verts = [v for v in bb.get_world_vertices(carla.Transform())]
        x_max, x_min, y_max, y_min = self.get_x_y_min_max(world_2_camera, verts)
        width = int(x_max) - int(x_min)
        height = int(y_max) - int(y_min)
        label = self.object_labels_dict[str(_object.type)]
        input_array = np.ones((height, width))*label
        new_matrix[int(y_min): int(y_max), int(x_min): int(x_max)] = input_array
        return new_matrix
    
    def check_if_object_is_in_front_range(
            self, _object):
        dist = _object.transform.location.distance(self.camera.get_transform().location)
        is_in_range = (dist < self.max_distance) and (dist > self.min_distance)
        if is_in_range:
            # Calculate the dot product between the forward vector
            # of the vehicle and the vector between the vehicle
            # and the bounding box. We threshold this dot product
            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
            forward_vec = self.camera.get_transform().get_forward_vector()
            ray = _object.transform.location - self.camera.get_transform().location
            return forward_vec.dot(ray) > 1
        return False
            
        
    def create_label_matrix(self):# -> list[np.ndarray, bool, np.ndarray]:
        img, new_matrix = self.get_image_and_create_copy_zeros_matrix()
        # Get the camera matrix 
        world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())
        objects = self.get_objects()
        exists_object = False
        for _object in objects:
            # Filter for distance from ego vehicle
            if self.check_if_object_is_in_front_range(_object):
                # Cycle through the vertices
                exists_object = True
                new_matrix = self.add_object_label_into_new_matrix(_object, world_2_camera, new_matrix)
        return new_matrix, exists_object, img