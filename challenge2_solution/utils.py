import cv2
import numpy as np
import base64
import pandas as pd
import cv2
import sqlite3




def convert_csv_to_sql():
    # Read in the CSV file and store it in a DataFrame
    df = pd.read_csv('Challenge2.csv')
    
    # Extract the depth values from the DataFrame
    depth = df['depth'].values
    
    # Remove the depth column from the DataFrame
    images = df.drop(columns=['depth']).values

    # Resize the images to a new size of 15x10 pixels and flatten them into 1-dimensional arrays
    resized_images = []
    for image in images:
        reshaped_image = image.reshape((20, 10))
        resized_image = cv2.resize(reshaped_image, (15, 10)).astype(np.uint8)
        # Uncomment this line to apply a color map to the image
        # resized_image = cv2.applyColorMap(resized_image, cv2.COLORMAP_JET)
        resized_images.append(resized_image.flatten())

    # Create a new DataFrame with the flattened images and their corresponding depth values
    resized_df = pd.DataFrame({'image': resized_images, 'depth': depth})

    # Connect to an SQLite database named 'resized_images.db'
    conn = sqlite3.connect('resized_images.db')

    # Create a table in the database if it doesn't already exist
    conn.execute('CREATE TABLE IF NOT EXISTS Images (id INTEGER PRIMARY KEY AUTOINCREMENT, depth REAL, image BLOB)')

    # Insert each row of the DataFrame into the 'Images' table as a new record
    for i, row in resized_df.iterrows():
        conn.execute('INSERT INTO Images (depth, image) VALUES (?, ?)', (row['depth'], row['image']))

    # Commit any changes made to the database and close the connection
    conn.commit()
    conn.close()


def apply_color_map(image_data):
    # Convert the 1-dimensional array of pixel values back into a 2D image array
    image_2d = np.asanyarray(image_data)
    image_2d = image_2d.reshape((15, 10)).astype(np.uint8)
    
    # Reshape the 2D image array to its original size of 15x10 pixels
    image_2d = image_2d.reshape((15, 10)).astype(np.uint8) 
    
    # Apply a color map to the image and return it as a new 2D image array
    colored_image = cv2.applyColorMap(image_2d, cv2.COLORMAP_JET)
    return colored_image


def convert_image_to_bytes(image_array):
    # Encode the 2D image array as a PNG image file in memory
    _, buffer = cv2.imencode('.png', image_array)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Convert the encoded image file to a base64 string and return it
    image_base64 = base64.b64encode(buffer).decode('utf-8') 
    return image_base64