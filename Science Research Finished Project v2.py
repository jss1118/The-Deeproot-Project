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

# Variable definitions
confidence = 0.5
disease = False

# Folder clearing
output_dir = '/Users/joshua.stanley/Desktop/Science Research/Model outputs/R-CNN/zoomed_bounding_boxes/test_dataset/'



leaf_model = tf.keras.models.load_model('/Users/joshua.stanley/Desktop/real2.keras')




class_names = ['Cassava__brown_streak_disease', 'Potato___Early_blight', 'Cassava__green_mottle', 'Blueberry___healthy', 'Cassava__healthy', 'Corn_(maize)___healthy', 'Cassava__mosaic_disease', 'Tomato___Target_Spot', 'Chili__healthy', 'Chili__leaf curl', 'Chili__whitefly', 'Chili__leaf spot', 'Potato___Late_blight', 'Chili__yellowish', 'Tomato___Late_blight', 'Coffee__cercospora_leaf_spot', 'Tomato___Tomato_mosaic_virus', 'Coffee__healthy', 'Coffee__red_spider_mite', 'Orange___Haunglongbing_(Citrus_greening)', 'Coffee__rust', 'Tomato___Leaf_Mold', 'Cucumber__diseased', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Cucumber__healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Gauva__diseased', 'Apple___Cedar_apple_rust', 'Gauva__healthy', 'Tomato___Bacterial_spot', 'Grape__black_measles', 'Grape___healthy', 'Jamun__diseased', 'Tomato___Early_blight', 'Jamun__healthy', 'Lemon__diseased', 'Grape___Esca_(Black_Measles)', 'Lemon__healthy', 'Raspberry___healthy', 'Mango__diseased', 'Tomato___healthy', 'Mango__healthy', 'Cherry_(including_sour)___healthy', 'Pomegranate__diseased', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Pomegranate__healthy', 'Apple___Apple_scab', 'Rice__brown_spot', 'Corn_(maize)___Northern_Leaf_Blight', 'Rice__healthy', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Rice__hispa', 'Soybean__rust', 'Peach___Bacterial_spot', 'Rice__leaf_blast', 'Pepper,_bell___Bacterial_spot', 'Rice__neck_blast', 'Tomato___Septoria_leaf_spot', 'Squash___Powdery_mildew', 'Soybean__caterpillar', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Soybean__diabrotica_speciosa', 'Apple___Black_rot', 'Soybean__downy_mildew', 'Apple___healthy', 'Soybean__mosaic_virus', 'Strawberry___Leaf_scorch', 'Soybean__powdery_mildew', 'Potato___healthy', 'Soybean__southern_blight', 'Soybean___healthy', 'Sugarcane__healthy', 'Sugarcane__red_rot', 'Sugarcane__red_stripe', 'Sugarcane__rust', 'Tea__algal_leaf', 'Tea__anthracnose', 'Tea__bird_eye_spot', 'Tea__brown_blight', 'Tea__healthy', 'Tea__red_leaf_spot', 'Wheat__brown_rust', 'Wheat__healthy', 'Wheat__septoria', 'Strawberry___healthy', 'Cassava__bacterial_blight', 'Peach___healthy', 'Pepper,_bell___healthy', 'Corn_(maize)___Common_rust_', 'Soybean__bacterial_blight', 'Sugarcane__bacterial_blight', 'Wheat__yellow_rust', 'Grape___Black_rot']

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
                        data = tf.image.resize(cropped_leaf, [256, 256])
                        data = np.expand_dims(data, axis=0)
                        prediction = leaf_model.predict(data, verbose=0)

                        
                        # Fixed preprocessing pipeline
                        predicted_class = np.argmax(prediction[0])
                        class_label = class_names[predicted_class]
                        print(class_label)
                        color = (0, 255, 0) if 'healthy' in class_label else (0, 0, 255)
                        label = f"{class_label} ({conf:.2f})"
                        print(class_label)
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
                        data = tf.image.resize(cropped_leaf, [256, 256])
                        data = np.expand_dims(data, axis=0)
                        prediction = leaf_model.predict(data)

                        
                        # Fixed preprocessing pipeline
                        
                        predicted_class = np.argmax(prediction[0])
                        class_label = class_names[predicted_class]

                        color = (0, 255, 0) if 'healthy' in class_label else (0, 0, 255)
                        label = f"{class_label} ({conf:.2f})"

                        print(class_label)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        save_path = os.path.join(output_dir, f"{label.replace(' ', '_')}.jpg")
                        cv2.imwrite(save_path, cropped_leaf)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
            self.video_label.configure(image=img)
            self.video_label.image = img
        self.sidebar_button_run.configure(state="Enabled", text="Run")    

if __name__ == "__main__":
    app = App()
    app.mainloop()
