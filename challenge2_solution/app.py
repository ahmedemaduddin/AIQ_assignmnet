from flask import Flask, request, jsonify
import sqlite3
import numpy as np
from utils import apply_color_map, convert_image_to_bytes, convert_csv_to_sql

"""
Assumption: The image is 200 pixels. We reshape the image to (20,10), resize it to 
(15,10) if we flatten it it will be 150 width as a vector. We flatten it -> store it 
when we retrieve it, reshape it to (15,10)--> apply color map. The restored image will 
be a 3 channel image after applying the color map. The retrieved image is a base64 
image.
"""

convert_csv_to_sql()

app = Flask(__name__)


# Retrieves images based on depth range
def get_images_by_depth(depth_min, depth_max):
    conn = sqlite3.connect('resized_images.db')
    cursor = conn.execute("SELECT * FROM Images WHERE depth BETWEEN ? AND ?", (depth_min, depth_max))
    images = []
    for row in cursor:
        depth = row[1]
        # Converts the image data to a NumPy array
        image = np.frombuffer(row[2], dtype=np.uint8) 
        images.append({'depth': depth, 'image': image.tolist()})
    conn.close()
    return images


@app.route('/images', methods=['GET'])
def fetch_images():
    # Retrieves depth_min and depth_max from query parameters
    depth_min = request.args.get('depth_min', type=float)
    depth_max = request.args.get('depth_max', type=float)
    
    # Fetches images based on the depth range
    images = get_images_by_depth(depth_min, depth_max)

    """
    After restoring the image --> apply color map --> convert it to bytes.
    """

    # Applies color map and converts the image to bytes before returning
    images = [{'depth': image['depth'], 'image': convert_image_to_bytes(apply_color_map(image['image']))} for image in images]
    
    return jsonify(images)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
