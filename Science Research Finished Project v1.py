 #imports

import tkinter as tk
import customtkinter
from PIL import Image, ImageTk
import cv2
import threading
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import customtkinter as ctk
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
from colorama import *
from ultralytics import YOLO
import numpy as np
import os
import glob
import io
from tensorflow import keras
from tkVideoPlayer import TkinterVideo

# variable definitions
videostatus = ''
confidence = 0.5
disease = False

#folder clearing, to prevent the model from assesing old photos
def delete_files():
    directory = '/Users/joshua.stanley/Desktop/Science Research/Model outputs/R-CNN/zoomed_bounding_boxes/test_dataset/'

    files = glob.glob(os.path.join(directory, '*'))
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
            print('Folder cleared')
delete_files()


# GUI-----------------------------------------------------------------------------------------------------------------
#initiating class
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        #setting appearence
        customtkinter.set_appearance_mode("System")
        customtkinter.set_default_color_theme("blue")

        #setting size
        self.title("Sci Research 1.0")
        self.geometry(f"{1100}x{580}")
        #initializing grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        #creating frames
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="SciResearch v1", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        #run button
        self.sidebar_button_run = customtkinter.CTkButton(self.sidebar_frame, command=self.run_button_event, text='Run')
        self.sidebar_button_run.grid(row=1, column=0, padx=20, pady=10)

        #tabviews for changing settings and model output

        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Output")
        self.tabview.add("Settings")
        self.tabview.tab("Output").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Settings").grid_columnconfigure(0, weight=1)
        #this is for inputting a video directory to assess
        self.path_input_button = customtkinter.CTkButton(self.sidebar_frame, text="Video path",
                                                           command=self.open_input_dialog_event)
        self.path_input_button.grid(row=2, column=0, padx=20, pady=(10, 10))
        #changing the confidence in yolov8 object detection
        self.confidence_slider = customtkinter.CTkSlider(self.tabview.tab('Settings'), from_=0, to=1, number_of_steps=100, command=self.confidence_function)
        self.confidence_slider.grid(row=0, column=0, pady=2, padx=0)
        self.confidence_label = customtkinter.CTkLabel(self.tabview.tab('Settings'), text=f'Confidence > {(confidence) * 100}%')
        self.confidence_label.grid(row=1, column=0, pady=0, padx=0)
        #displayng model output
        self.output_label = customtkinter.CTkLabel(self.tabview.tab('Output'), text='N/A')
        self.output_label.grid(row=0, column=0, pady=3, padx=0)
        # Define switch_var as an instance attribute
        self.switch_var = customtkinter.StringVar(value="on")
        self.input_switch = customtkinter.CTkSwitch(self.sidebar_frame, text='Use custom image path', variable=self.switch_var, onvalue='on', offvalue='off')
        self.input_switch.grid(row=3, column=0)


        #displaying whether the video was found
        self.videostatus_label = customtkinter.CTkLabel(self.sidebar_frame, text=videostatus)
        self.videostatus_label.grid(row=4, column=0)
        self.video_label = customtkinter.CTkLabel(self, text='')
        self.video_label.grid(row=0, column=1, padx=(20, 10), pady=(10, 10), sticky="nsew")
        global total_progressbar
        self.total_progressbar = customtkinter.CTkProgressBar(self.tabview.tab('Output'), mode='determinate')
        self.total_progressbar.grid(row=1, column=0)
        self.total_progressbar_label = customtkinter.CTkLabel(self.tabview.tab('Output'), text=f'Model Progress: N/A')
        self.total_progressbar_label.grid(row=2, column=0)
        #radios
        self.radio_var = tk.IntVar(value=0)
        self.mode_radio_button_1 = customtkinter.CTkRadioButton(master=self.tabview.tab('Settings'), variable=self.radio_var, value=0, text='Live')
        self.mode_radio_button_1.grid(row=4, column=1, pady=10, padx=20, sticky="n")
        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.tabview.tab('Settings'), variable=self.radio_var, value=1, text='File')
        self.radio_button_2.grid(row=5, column=1, pady=10, padx=20, sticky="n")
        #disease output

        self.disease_label = customtkinter.CTkLabel(self.tabview.tab('Output'), text=f'Disease detected: {disease}')
        self.disease_label.grid(row=2, column=0, pady=2)
        #whether the model started label
        self.processing_label = customtkinter.CTkLabel(self.tabview.tab('Output'), text=f'Proccess: ')
        self.processing_label.grid(row=3, column=0)
        #initating video feed
        self.cap = cv2.VideoCapture(0)
        self.show_camera_feed()
        #initiating videoplayer for displaying results
        self.videoplayer = TkinterVideo(self, scaled=True)
        
        
        

        #END OF GUI INTERFACE CODE------------------------------------------








        #BACKEND------------------------------------------







    #this is the image path command

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in image path:", title="CTkInputDialog")
        global obj_image_path
        obj_image_path = dialog.get_input()
    #showing camera feed command
    def show_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video_label.configure(image=img)
            self.video_label.image = img
        self.video_label.after(10, self.show_camera_feed)

    #run button command
    def run_button_event(self):

        self.sidebar_button_run.configure(state="disabled", text="Running...")

        # Start recording video
        self.record_video()
    #command to the confidence slider to acc change it
    def confidence_function(self, value):
        global confidence
        confidence = float(value)
        self.confidence_label.configure(text=f'Confidence > {confidence * 100:.1f}%')
        print(f'Confidence > {confidence * 100:.1f}%')
    #object detection function
    
    
    
    
    
    
    
    
    
    
    
    
    
    def object_detect_leaf(self, callback):
        try:    
            def run():
                #initating model, turning off print statements
                model_path = '/Volumes/SanDisk SSD/detect/train32/weights/best.pt'
                model = YOLO(model_path)
                model.overrides['verbose'] = False  # Suppress logging in console during inference
                
                
                #setting some variables
                input_video_path = ''
                last_max_bounding_number = 0
                #if - else statement for whether or not the model uses live feed or file
                # Access switch_var using app instance
                
                
                if app.switch_var.get() == 'on':
                    global videostatus
                    #setting variables
                    input_video_path = f'/Users/joshua.stanley/Desktop/{str(obj_image_path)}'
                    self.video_label.pack_forget()
                    self.videoplayer.load(f'/Users/joshua.stanley/Desktop/{str(obj_image_path)}')
                    #making videplayer appear
                    self.videoplayer.grid(row=0, column=1, padx=(20, 10), pady=(10, 10), sticky="nsew")
                    videostatus = 'Video found.'
                    self.videoplayer.play() # play the video
                    #letting user know file was found
                    self.videostatus_label.configure(text=videostatus)

                else:
                    input_video_path = '/Users/joshua.stanley/my_video.mp4'
                    self.videoplayer.pack_forget()
                    #making videofeed appear
                    self.video_label.grid(row=0, column=1, padx=(20, 10), pady=(10, 10), sticky="nsew")
                    videostatus = 'Video recording started'
                    self.videostatus_label.configure(text=videostatus)
                
                output_video_path = '{}_out.mp4'.format(input_video_path)
                
                
                #detection parameters for video 
                video_capture = cv2.VideoCapture(input_video_path)
                frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(video_capture.get(cv2.CAP_PROP_FPS))
                total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                
                
                #file type setting
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

                bounding_boxes_detected = False
                frame_count = 0

                while video_capture.isOpened():
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model(frame)[0]
                    #looking for bounding boxes
                    bounding_boxes = []
                    for result in results.boxes.data.tolist():
                        x1, y1, x2, y2, conf, cls = result[:6]
                        #disregarding low conf detections
                        if conf >= confidence:
                            bounding_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                            label = f'{conf:.2f}'
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    #setting results to variables
                    if not bounding_boxes_detected and len(bounding_boxes) > last_max_bounding_number:
                        last_max_bounding_number = len(bounding_boxes)
                        bounding_boxes_detected = True
                        photo_path = os.path.expanduser("~/Desktop/full_frame.jpg")
                        cv2.imwrite(photo_path, frame)
                        #zooming into detections for classificaitons
                        zoomed_dir = os.path.expanduser("~/Desktop/Science Research/Model outputs/R-CNN/zoomed_bounding_boxes/test_dataset")
                        os.makedirs(zoomed_dir, exist_ok=True)

                        for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
                            cropped_image = rgb_frame[y1:y2, x1:x2]  # Use RGB frame to avoid bounding boxes in the cropped image
                            zoomed_image_path = os.path.join(zoomed_dir, f"box_{i+1}.jpg")
                            cv2.imwrite(zoomed_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                    
                    
                    
                    out_video.write(frame)
                    frame_count += 1
                    #displaying on gui
                    
                    
                    self.processing_label.configure(text=f'Processed frame {frame_count}/{total_frames}')
                    global detect_complete_percentage
                    global detect_step_percentage
                    
                    
                    detect_complete_percentage = int(frame_count) / int(total_frames)
                    detect_step_percentage = detect_complete_percentage * 80
                    self.total_progressbar.set((round(detect_step_percentage, 2)) / 100)
                    self.total_progressbar_label.configure(self.tabview.tab('Output'), text=f'Model Progress: {int(detect_complete_percentage) * 100} %')
                
                #error handling for no detections being found
                
                
                if last_max_bounding_number == 0:
                    print(Fore.RED + "0 BBOX created. Please adjust camera or video angle where surface leaves are visible")
                video_capture.release()
                out_video.release()
                cv2.destroyAllWindows()
                print(f'Output video saved to {output_video_path}')
                self.video_label.pack_forget()
                self.videoplayer.load(output_video_path)
                self.videoplayer.grid(row=0, column=1, padx=(20, 10), pady=(10, 10), sticky="nsew")
                self.videoplayer.play() # play the video
                videostatus = 'Video found.'
                self.videostatus_label.configure(text='Displaying detection')
                
                
                
                if callback:
                    callback()  # Start the next thread in the sequence
            threading.Thread(target=run, daemon=True).start()
        except:
            print(Fore.RED + 'Program will not run normally, SSD is not found.')
            print(Fore.WHITE + '')
# DISEASE DETECTION-------------------------------------------------



    def disease_model_run(self):
        def run():
            #setting model from file
            global class_number, class_names, class_predictions
            model = keras.models.load_model('/Users/joshua.stanley/Desktop/real.keras')
            test_dataset = '/Users/joshua.stanley/Desktop/Science Research/Model outputs/R-CNN/zoomed_bounding_boxes/'
            test_data = tf.keras.preprocessing.image_dataset_from_directory(
                test_dataset,
                image_size=(256, 256)
            )
            #defining classes
            class_predictions = {}
            class_number = 0
            class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
            
            
            #loop
            for images, labels in test_data.take(1):
                predictions = model.predict(images)
                
                # Ensure you iterate over each prediction properly
                for i in range(len(predictions)):
                    global disease
                    predicted_class = np.argmax(predictions[i])

                    # Ensure the predicted class is within the bounds of the class_names list
                    if 0 <= predicted_class < len(class_names):
                        class_number += 1
                        class_result = {class_number: class_names[predicted_class]}
                        class_predictions.update(class_result)

                        if 'healthy' not in class_names[predicted_class]:
                            disease = True
                    else:
                        print(f"Warning: Predicted class index {predicted_class} is out of range.")
            #results
            print(f'Disease detected: {disease}')
            self.disease_label.configure(text=f'Disease detected: {disease}, \n Type:  {class_predictions}')
            print(f'Data: {class_predictions}')
            
            
            self.total_progressbar.set(1)
            self.sidebar_button_run.configure(state="enabled", text="Run")
        
        
        threading.Thread(target=run, daemon=True).start()
    def record_video(self, duration=10, output_filename="my_video.mp4"):
        def run():
            #initiating videofeed
            if self.switch_var.get() == 'off':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

                start_time = time.time()
                frames_written = 0
                #displaying video
                while time.time() - start_time < duration:
                    ret, frame = self.cap.read()
                    if ret:
                        out.write(frame)
                        frames_written += 1
                    else:
                        print("Failed to read frame from camera.")
                        break

                out.release()
                

                if frames_written == 0:
                    print("No frames were written to the video. Check your camera and codec settings.")

                # Start the object detection after recording
            self.object_detect_leaf(callback=self.start_disease_detection)
        #multithreading to prevent gui from freezing
        threading.Thread(target=run, daemon=True).start()
    #initating thread
    def start_disease_detection(self):
        self.disease_model_run()

#starting code
if __name__ == "__main__":
    app = App()
    app.mainloop()
#done!