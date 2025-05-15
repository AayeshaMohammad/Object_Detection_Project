import cv2
import numpy as np
import tensorflow as tf


model_path = r'C:\Users\moham\OneDrive\Documents\saved_model'  # Update this path
image_path = r'C:\Users\moham\OneDrive\Documents\traffic.jpg'  # Updated image path


model = tf.saved_model.load(model_path)


image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
else:
    print("Image loaded successfully!")


    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]


    detections = model(input_tensor)


    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']
    classes = detections['detection_classes'].astype(np.int64)


    for i in range(num_detections):
        if scores[i] > 0.5:  # Confidence threshold
            box = boxes[i]
            h, w, _ = image.shape
            y_min, x_min, y_max, x_max = box

            start_point = (int(x_min * w), int(y_min * h))
            end_point = (int(x_max * w), int(y_max * h))


            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)


cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
