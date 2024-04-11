import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
import time
import os
import datetime as datetime
from multiprocessing import Process, Event, Pool

# Setup GPIO for Radar/IR detector
GPIO.setmode(GPIO.BCM)
GPIO.setup(4, GPIO.IN)

# Path to yolov5 model
model_path = 'yolo.tflite'
thresh = 0.7

# Event initialization
object_detected_event = Event()
processing_done_event = Event()

# Class labels for yolov5
class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def gpio_callback(channel):
    if GPIO.input(channel):
        print("Radar detected object")
        object_detected_event.set()
    else:
        print("Object no longer detected - clearing events. Back to low power.")
        object_detected_event.clear()
        processing_done_event.set()

GPIO.add_event_detect(4, GPIO.BOTH, callback=gpio_callback, bouncetime=200)

def save_frame(frame, detected_classes):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    classes_str = "_".join(detected_classes) if detected_classes else "no_detection"
    filename = f"detection_{classes_str}_{timestamp}.jpg"
    filepath = os.path.join("/home/dietpi/", filename)
    cv2.imwrite(filepath, frame)
    print(f"Image saved: {filename}")

def get_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def recognize_frame(frame, model_path, class_labels):
    print(f"Run interpretation")
    interpreter = get_interpreter(model_path)
    input_data = preprocess_frame(frame, interpreter.get_input_details()[0]['shape'])
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    detections = interpret_output(output_data, frame.shape, class_labels)
    detected_info = [(det[0], det[1]) for det in detections] if detections else []
    if detected_info:
        print(f"Detected: {detected_info}")
    return [det[0] for det in detections]

def preprocess_frame(frame, input_shape):
    print(f"Preprocessing captured frames")
    frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))  # Width, Height for the model
    frame_normalized = frame_resized / 255.0
    input_data = np.expand_dims(frame_normalized, axis=0).astype(np.float32)
    return input_data

def interpret_output(output_data, frame_shape, class_labels, score_threshold=thresh):
    detections = []
    for detection in output_data[0]:
        cx, cy, w, h, score, *class_probs = detection
        if score >= score_threshold:
            class_id = np.argmax(class_probs)
            class_name = class_labels[class_id]
            x_min = int((cx - w / 2) * frame_shape[1])
            y_min = int((cy - h / 2) * frame_shape[0])
            x_max = int((cx + w / 2) * frame_shape[1])
            y_max = int((cy + h / 2) * frame_shape[0])
            area = (x_max - x_min) * (y_max - y_min)
            detections.append((class_name, score, (x_min, y_min, x_max, y_max), area))
    detections.sort(key=lambda x: x[3], reverse=True)
    unique_detections = []
    for det in detections:
        if det[0] not in [d[0] for d in unique_detections]:
            unique_detections.append(det)
            if len(unique_detections) == 2:  # Stop after finding 2 unique class detections
                break
    return unique_detections

def frame_capture_and_processing(object_detected_event, processing_done_event, model_path, class_labels):
    print("Capture open and processing start.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error:    Camera could not be opened.")
        return
    
    with Pool(processes=os.cpu_count()) as pool:  # This will use as many processes as there are CPUs
        while not processing_done_event.is_set():
            if object_detected_event.is_set():
                ret, frame = cap.read()
                if ret:
                    interpreter = get_interpreter(model_path)
                    input_data = preprocess_frame(frame, interpreter.get_input_details()[0]['shape'])
                    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

                    detections = pool.apply_async(interpret_output, (output_data, frame.shape, class_labels)).get()
                    
                    detected_classes = [det[0] for det in detections] if detections else []
                    if detected_classes:
                        save_frame(frame, detected_classes)
                        print("Frame with successful detection saved.")
                    else:
                        print("No recognised objects meeting threshold.")
    cap.release()

def prepare_recognition_summary(detections):
    if not detections:
        return "No objects recognised meeting threshold."
    
    # Count the occurrences of each class in the detections
    class_counts = {}
    for detection in detections:
        class_name = detection['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # summarise
    summary_parts = []
    for class_name, count in class_counts.items():
        part = f"{count} {class_name}(s)" if count > 1 else f"{count} {class_name}"
        summary_parts.append(part)
    
    summary_message = "Detected: " + ", ".join(summary_parts) + "."
    return summary_message

def end_recognition_event(detections):

    summary_message = prepare_recognition_summary(detections)
    print(summary_message)
    
    object_detected_event.clear()
    processing_done_event.clear()


if __name__ == '__main__':
    try:
        processing_thread = None  # Initialize processing_thread outside the try block
        while True:
            object_detected_event.wait()  # Wait for an object to be detected

            # Frame capture and processing
            processing_thread = Process(target=frame_capture_and_processing, args=(object_detected_event, processing_done_event, model_path, class_labels))
            processing_thread.start()

            processing_thread.join()

            # Reset events for the next detection
            object_detected_event.clear()
            processing_done_event.clear()

            print("System reset, waiting for next object detection...")
    except KeyboardInterrupt:
        print("Shutting down.")
        GPIO.cleanup()
    finally:
        if processing_thread and processing_thread.is_alive():
            processing_thread.terminate()
            processing_thread.join()