# Import necessary libraries and modules
import os
from flask import Flask, request, jsonify
import uuid
import json
from process_image import CirclesDetection

app = Flask(__name__)

MODEL_PATH = os.path.join(os.getcwd(), 'coin.onnx')
# Define the path to the ONNX model for object detection
MODEL_PATH = os.path.join(os.getcwd(), 'coin.onnx')  
detection = CirclesDetection(MODEL_PATH)

# Define the upload folder for storing uploaded images
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
# Define a route for uploading images
def upload_image():
    file = request.files['image']
    # Retrieve the image file from the request
    if file:
    
        image_id = str(uuid.uuid1())
        # Generate a unique ID for the uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, image_id + '.png')
        
        # Define the path to save the uploaded image
        file.save(image_path)
        
        # Save the uploaded image to the designated folder

        detection.circular_objects(image_id)  
        detection.circular_objects(image_id)
        # Call the circular_objects method from the CirclesDetection class to detect objects in the image
        detection.circular_objects(image_id)  
        return jsonify({'image_id': image_id})
        # Return a JSON response with the unique ID of the uploaded image
    return "No file uploaded", 400

@app.route('/get_circular_objects/<image_id>', methods=['GET'])
def get_circular_objects(image_id):
    image_json_path = os.path.join(UPLOAD_FOLDER, f'{image_id}.json')
    circular_objects = None
    if os.path.exists(image_json_path):
        with open(image_json_path, 'r') as f:
            circular_objects = json.load(f)['objects'].values()
        objects = []
        for obj in circular_objects:
            objects.append({"objec_id": obj['obj_id'],
                            "bounding_box": obj['bounding_box']
                            })

        return json.dumps(objects, indent=2)
    return "No circular objects found for the image", 404

@app.route('/object/<object_id>', methods=['GET'])
def get_circular_object(object_id):
    master_json_path = os.path.join(os.getcwd(), 'master.json')
    if os.path.exists(master_json_path):
        with open(master_json_path, 'r') as f:
            circular_objects = json.load(f)

        if object_id in circular_objects:
            return jsonify({
                'bounding_box': circular_objects[object_id]['bounding_box'],
                'centroid': circular_objects[object_id]['centroid'],
                'radius': circular_objects[object_id]['radius'],
            })
    return "Object not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
