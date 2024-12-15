import tkinter as tk
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import os
import datetime
# Variable definitions
confidence = 0.5
disease = False

# Folder clearing
output_dir = '/Users/joshua.stanley/Desktop/Science Research/Model outputs/R-CNN/zoomed_bounding_boxes/test_dataset/'



leaf_model = tf.keras.models.load_model('/Users/joshua.stanley/Desktop/newmodel.keras')




class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
# GUI
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("System")
        customtkinter.set_default_color_theme("blue")

        self.title("Sci Research 1.0")
        self.geometry("1100x580")
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="SciResearch v1", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_run = customtkinter.CTkButton(self.sidebar_frame, command=self.run_button_event, text='Run')
        self.sidebar_button_run.grid(row=1, column=0, padx=20, pady=10)

        self.tabview = customtkinter.CTkTabview(self, width=250)
        self.tabview.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.tabview.add("Output")
        self.tabview.add("Settings")
        self.tabview.tab("Output").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Settings").grid_columnconfigure(0, weight=1)

        self.confidence_slider = customtkinter.CTkSlider(self.tabview.tab('Settings'), from_=0, to=1, number_of_steps=100, command=self.confidence_function)
        self.confidence_slider.grid(row=0, column=0, pady=2, padx=0)
        self.confidence_label = customtkinter.CTkLabel(self.tabview.tab('Settings'), text=f'Confidence > {(confidence) * 100}%')
        self.confidence_label.grid(row=1, column=0, pady=0, padx=0)

        self.output_label = customtkinter.CTkLabel(self.tabview.tab('Output'), text='N/A')
        self.output_label.grid(row=0, column=0, pady=3, padx=0)

        self.video_label = customtkinter.CTkLabel(self, text='')
        self.video_label.grid(row=0, column=1, padx=(20, 10), pady=(10, 10), sticky="nsew")

        self.radio_var = tk.IntVar(value=0)
        self.radio_button_live = customtkinter.CTkRadioButton(self.sidebar_frame, text="Live Detection", variable=self.radio_var, value=0)
        self.radio_button_live.grid(row=2, column=0, padx=20, pady=(10, 0))
        self.radio_button_file = customtkinter.CTkRadioButton(self.sidebar_frame, text="File Detection", variable=self.radio_var, value=1)
        self.radio_button_file.grid(row=3, column=0, padx=20, pady=(10, 0))

        self.file_path = None
        self.file_button = customtkinter.CTkButton(self.sidebar_frame, text="Select File", command=self.select_file)
        self.file_button.grid(row=4, column=0, padx=20, pady=(10, 0))

    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
        )
        if self.file_path:
            print(f"Selected file: {self.file_path}")

    def run_button_event(self):
        self.sidebar_button_run.configure(state="disabled", text="Running...")
        if self.radio_var.get() == 0:
            threading.Thread(target=self.live_detection_thread, daemon=True).start()
        else:
            threading.Thread(target=self.file_detection_thread, daemon=True).start()

    def confidence_function(self, value):
        global confidence
        confidence = float(value)
        self.confidence_label.configure(text=f'Confidence > {confidence * 100:.1f}%')
        print(f'Confidence > {confidence * 100:.1f}%')

    def live_detection_thread(self):
        model = YOLO('/Users/joshua.stanley/Desktop/train32/weights/best.pt')
        model.overrides['verbose'] = False
        cap = cv2.VideoCapture(0)
        cap.set(3, 1000)
        cap.set(4, 500)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame for zoom-out effect (scale factor < 1)
            scale_factor = 0.5  # Adjust this factor as needed (e.g., 0.5 for 50% size)
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

            results = model(frame)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]

                    if conf >= confidence:
                        cropped_leaf = frame[y1:y2, x1:x2]
                        if cropped_leaf.size == 0:
                            print("Cropped leaf is empty! Check bounding box coordinates.")
                        else:
                            print("Cropped leaf shape:", cropped_leaf.shape)
                        cropped_leaf = cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # Unique timestamp
                        save_path = os.path.join(output_dir, f"leaf_{timestamp}.jpg")
                        cv2.imwrite(save_path, cv2.cvtColor(cropped_leaf, cv2.COLOR_RGB2BGR))
                        print(f"Saved cropped leaf to {save_path}")
                        data = tf.image.resize(cropped_leaf, [128, 128])
                        data = np.expand_dims(data, axis=0)
                        prediction = leaf_model.predict(data, verbose=0)

                        
                        # Fixed preprocessing pipeline
                        confidence_threshold = 0.6  # Set your desired confidence threshold

                        predicted_class = np.argmax(prediction[0])
                        conf = prediction[0][predicted_class]  # Get confidence for the predicted class

                        if conf < confidence_threshold:  # Check if confidence is below the threshold
                            class_label = "get closer to leaf"
                            color = (255, 255, 0)  # Yellow for low confidence
                            label = class_label
                        else:
                            class_label = class_names[predicted_class]
                            color = (0, 255, 0) if 'healthy' in class_label else (0, 0, 255)
                            label = f"{class_label} ({conf:.2f})"

                        print(class_label)

                        # Draw rectangle and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
            self.video_label.configure(image=img)
            self.video_label.image = img

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.sidebar_button_run.configure(state="Enabled", text="Run")
    def file_detection_thread(self):
        if not self.file_path:
            print("No file selected for detection.")
            self.sidebar_button_run.configure(state="enabled", text="Run")
            return

        model = YOLO('/Users/joshua.stanley/Desktop/train32/weights/best.pt')
        model.overrides['verbose'] = False
        cap = cv2.VideoCapture(self.file_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            scale_factor = 0.5
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

            results = model(frame)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]

                    if conf >= confidence:
                        cropped_leaf = frame[y1:y2, x1:x2]
                        if cropped_leaf.size == 0:
                            print("Cropped leaf is empty! Check bounding box coordinates.")
                        else:
                            print("Cropped leaf shape:", cropped_leaf.shape)
                        cropped_leaf = cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB)
                        data = tf.image.resize(cropped_leaf, [128, 128])
                        data = np.expand_dims(data, axis=0)
                        prediction = leaf_model.predict(data)

                        
                        # Fixed preprocessing pipeline
                        
                        confidence_threshold = 0.6  # Set your desired confidence threshold

                        predicted_class = np.argmax(prediction[0])
                        conf = prediction[0][predicted_class]  # Get confidence for the predicted class

                        if conf < confidence_threshold:  # Check if confidence is below the threshold
                            class_label = "get closer to leaf"
                            color = (255, 255, 0)  # Yellow for low confidence
                            label = class_label
                        else:
                            class_label = class_names[predicted_class]
                            color = (0, 255, 0) if 'healthy' in class_label else (0, 0, 255)
                            label = f"{class_label} ({conf:.2f})"

                        print(class_label)

                        # Draw rectangle and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
            self.video_label.configure(image=img)
            self.video_label.image = img
        self.sidebar_button_run.configure(state="Enabled", text="Run")    

if __name__ == "__main__":
    app = App()
    app.mainloop()
