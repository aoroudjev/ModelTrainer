import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [640, 640])
    img = tf.cast(img, tf.uint8)  # Ensure the type is uint8
    return img.numpy()  # Convert to numpy array


# Replace 'path_to_your_image.jpg' with your actual image path
image = preprocess_image('./dataset/images/Hand_0000038.jpg')

# Load the TFLite model and allocate tensors
interpreter = hub.load("https://www.kaggle.com/models/tensorflow/efficientdet/frameworks/TensorFlow2/variations/d7/versions/1")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set the tensor to point to the input data to be inferred
interpreter.set_tensor(input_details[0]['index'], [image])

# Run the inference
interpreter.invoke()

# Extract the output tensors
boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes
classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class indices
scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
count = int(interpreter.get_tensor(output_details[3]['index'])[0])  # Number of detections

def draw_boxes(image, boxes, class_names, scores, count):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for i in range(count):
        ymin, xmin, ymax, xmax = boxes[i]
        (left, right, top, bottom) = (xmin * 640, xmax * 640, ymin * 640, ymax * 640)

        width, height = right - left, bottom - top
        rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Check if class index is within the range of class_names
        class_index = int(classes[i])
        if class_index < len(class_names):
            label = class_names[class_index]
        else:
            label = 'Unknown'  # Use a placeholder for unknown class indices

        ax.text(left, top, f'{label}: {scores[i]:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()

draw_boxes(image, boxes, class_names, scores, count)