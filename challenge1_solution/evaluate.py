import sys
import os
import warnings
import onnxruntime as ort
import cv2
from tqdm import tqdm
from process_image import CirclesDetection
import numpy as np
# Disable logging for ONNX Runtime
ort.set_default_logger_severity(4)  # Disable most logs

# Suppress OpenCV warnings
sys.stderr = open(os.devnull, 'w')

# Suppress all Python warnings
warnings.filterwarnings("ignore")

class CircleDetectionEvaluator:
    def __init__(self, model_path, val_images_path, val_labels_path):
        self.detection_model = CirclesDetection(model_path)
        self.val_images_path = val_images_path
        self.val_labels_path = val_labels_path

    def load_ground_truth(self, label_path, img_shape):
        """Parse YOLO label file to get bounding boxes."""
        with open(label_path, 'r') as f:
            lines = f.readlines()

        height, width = img_shape[:2]
        gt_data = []
        for line in lines:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

            x_center *= width
            y_center *= height
            bbox_width *= width
            bbox_height *= height

            x_min = x_center - (bbox_width / 2)
            y_min = y_center - (bbox_height / 2)
            bbox = [x_min, y_min, bbox_width, bbox_height]
            gt_data.append({'bbox': bbox, 'class_id': int(class_id)})
        return gt_data

    def predict_circles(self, image_path):
        mask, circles = self.detection_model.detect_circles(image_path)
        predicted_data = []
        if circles:
            for circle in circles:
                # Convert circle to bounding box [x_min, y_min, width, height]
                bbox = [circle[0] - circle[2], circle[1] - circle[2], circle[2] * 2, circle[2] * 2]
                predicted_data.append({'bbox': bbox})
        return predicted_data

    def compute_iou(self, boxA, boxB):
        """Compute IoU between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def evaluate_model(self):
        results = []
        images = os.listdir(self.val_images_path)

        for image_file in tqdm(images):
            image_path = os.path.join(self.val_images_path, image_file)
            label_path = os.path.join(self.val_labels_path, os.path.splitext(image_file)[0] + '.txt')

            # Load ground truth and predictions
            img = cv2.imread(image_path)
            gt_data = self.load_ground_truth(label_path, img.shape)
            predicted_data = self.predict_circles(image_path)

            # Compute IoU for each ground truth and prediction pair
            image_results = []
            for gt in gt_data:
                best_iou = 0
                for pred in predicted_data:
                    iou = self.compute_iou(gt['bbox'], pred['bbox'])
                    best_iou = max(best_iou, iou)
                image_results.append({'iou': best_iou})
            
            results.append({'image_id': image_file, 'results': image_results})
        return results

    def compute_average_iou(self, evaluation_results):
        all_ious = [result['iou'] for image_result in evaluation_results for result in image_result['results']]
        return np.mean(all_ious)

if __name__ == '__main__':
    model_path = 'coin.onnx'
    val_images_path = './coint_dataset_yolo/val/images'
    val_labels_path = './coint_dataset_yolo/val/labels'

    evaluator = CircleDetectionEvaluator(model_path, val_images_path, val_labels_path)
    evaluation_results = evaluator.evaluate_model()
    average_iou = evaluator.compute_average_iou(evaluation_results)
    print(f"Average IoU: {average_iou}")

