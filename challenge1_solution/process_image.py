import os
import json
import uuid
import numpy as np
import cv2
from ultralytics import YOLO

class CirclesDetection:
    def __init__(self, model_path, upload_folder='uploads/', master_json='master.json'):
        # Load the pretrained YOLO model
        self.model = YOLO(model_path, task='detect')
        
        self.upload_folder = upload_folder
        self.master_json = master_json
        self.__circular_objects = {}  # Stores object data globally
        self.__image_data = {}  # Stores objects related to each image
        
        # Ensure the upload folder exists
        os.makedirs(self.upload_folder, exist_ok=True)
        
        # If master JSON doesn't exist, create it
        if not os.path.exists(self.master_json):
            with open(self.master_json, 'w') as f:
                json.dump({}, f)

    def detect_circles(self, image_path):
        # Run inference on the image
        results = self.model([image_path])
        
        # Process results for the first image in the batch
        result = results[0]
        
        # Extract bounding boxes and masks (if available)
        boxes = result.boxes
        masks = result.masks
        
        circles = []
        
        # Load the image to get its dimensions
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]  # Get image dimensions (height, width)

        if len(boxes) > 0:
            for box in boxes:
                box = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = box

                # Calculate centroid and radius
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                radius = min(x2 - x1, y2 - y1) / 2

                circles.append((center_x, center_y, radius))

        # Create a mask for the detected circles
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for (x, y, r) in circles:
            cv2.circle(mask, (int(x), int(y)), int(r), (255), -1)

        return mask, circles

    def circular_objects(self, image_id):
        image_path = os.path.join(self.upload_folder, image_id + '.png')
        mask, circles = self.detect_circles(image_path)

        if circles:
            # Store object data in both the master and image-specific JSON files
            image_object_ids = []
            for (x, y, r) in circles:
                obj_id = str(uuid.uuid4())  # Generate a unique object ID
                object_data = {
                    'obj_id': obj_id,
                    'bounding_box': [int(x - r), int(y - r), int(x + r), int(y + r)],
                    'centroid': (int(x), int(y)),
                    'radius': int(r),
                    'area': np.pi * int(r) * int(r)
                }
                self.__circular_objects[obj_id] = object_data  # Store object details globally
                image_object_ids.append(obj_id)  # Store object IDs for the current image

            # Save this image's object IDs to its JSON file
            image_json_path = os.path.join(self.upload_folder, f'{image_id}.json')
            
            with open(image_json_path, 'w') as f:
                json.dump({ 'objects': self.__circular_objects}, f, indent=4)

            # Update the master JSON file with object details
            with open(self.master_json, 'r') as f:
                master_data = json.load(f)

            for obj in self.__circular_objects.values():
                master_data[obj['obj_id']] = obj  # Add object data to the master JSON

            with open(self.master_json, 'w') as f:
                json.dump(master_data, f, indent=4)

    def get_circular_objects(self, image_id=None):
        """Return all circular objects for a specific image, or all objects if no image_id is given"""
        if image_id:
            # Fetch object IDs for a specific image
            image_json_path = os.path.join(self.upload_folder, f'{image_id}.json')
            if os.path.exists(image_json_path):
                with open(image_json_path, 'r') as f:
                    return json.load(f)
            else:
                return "No data found for this image.", 404
        else:
            # Return all circular objects in the master JSON
            with open(self.master_json, 'r') as f:
                return json.load(f)
