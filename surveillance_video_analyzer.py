import os
import sqlite3
import cv2
import csv
import time
import threading
from queue import Queue
from PIL import Image
import torch
import datetime
import logging
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

# Set up the device for TensorFlow
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration and modify if necessary
config = AutoConfig.from_pretrained("kndrvitja/florence-SPHAR-finetune-2", trust_remote_code=True)
if config.vision_config.model_type != 'davit':
    config.vision_config.model_type = 'davit'

# Load the model and processor
model = AutoModelForCausalLM.from_pretrained("kndrvitja/florence-SPHAR-finetune-2", config=config, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("kndrvitja/florence-SPHAR-finetune-2", trust_remote_code=True)

# Define paths
video_folder_path = '/teamspace/studios/this_studio/content/drive/MyDrive/Florence_2_video_analytics'
frame_save_path = '/teamspace/studios/this_studio/Florence_2_video_analytics/frames'
if not os.path.exists(frame_save_path):
    os.makedirs(frame_save_path)

# Queue for thread communication
frame_queue = Queue()

# Database setup function
def setup_database(db_file_path):
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS frame_data (
        timestamp TEXT,
        frame_path TEXT,
        result TEXT
    )
    ''')
    conn.commit()
    conn.close()

# Configure logging
logging.basicConfig(filename='/teamspace/studios/this_studio/Florence_2_video_analytics/debug.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

def extract_frames(video_path, interval=1):
    """ Extracts frames from a video file every 'interval' seconds. """
    for video_file in video_files:
        try:
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                current_time = frame_number / fps
                if current_time >= interval:
                    frame_path = os.path.join(frame_save_path, f"frame_{int(current_time)}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_queue.put(frame_path)
                    #logging.info(f"Extracted frame at {frame_path}")
                    frame_number += int(fps * interval)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                else:
                    frame_number += 1
            cap.release()
        except Exception as e:
            logging.error(f"Error extracting frames from {video_file}: {e}")

# Function to run the model on an example
def run_example(task_prompt, text_input, image):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    # Tokenize inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    # Generate output
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    # Ensure parsed_answer is a string
    if isinstance(parsed_answer, dict):
        parsed_answer = str(parsed_answer)
    return parsed_answer

def process_frames(frame_queue, db_file_path):
    setup_database(db_file_path)
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    while True:
        frame_path = frame_queue.get()
        if frame_path is None:
            break

        try:
            # Load and preprocess the image
            image = Image.open(frame_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Call the run_example function
            answer = run_example(
                task_prompt="<SURVEILLANCE>",
                text_input='Given the surevillance image,create a detailed annotation capturing dynamic elements, individualls, actions, interactions,notable objects and events.The annotation should provide insights in to observed behaviours and situations in 2,3 lines',
                image=image
            )

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = f"Processed frame at {frame_path}: {answer}"
            cursor.execute("INSERT INTO frame_data (timestamp, frame_path, result) VALUES (?, ?, ?)",
                           (timestamp, frame_path, result))
            conn.commit()
            logging.info(f"Processed frame at {frame_path}: {answer}")

            cursor.execute("SELECT * FROM frame_data WHERE frame_path = ?", (frame_path,))
            data = cursor.fetchone()
            logging.info(f"Data verification for {frame_path}: {data}")


        except Exception as e:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_msg = f"Error processing {frame_path}: {e}"
            cursor.execute("INSERT INTO frame_data (timestamp, frame_path, result) VALUES (?, ?, ?)",
                           (timestamp, frame_path, error_msg))
            conn.commit()
            logging.error(error_msg)

    conn.close()

# List all video files
video_files = [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]

# Set up the database before starting processing
db_file_path = '/teamspace/studios/this_studio/Florence_2_video_analytics/Florence_2_video_analytics.db'
setup_database(db_file_path)

# Start threads
threading.Thread(target=extract_frames, args=(video_files,)).start()
threading.Thread(target=process_frames, args=(frame_queue, db_file_path)).start()
