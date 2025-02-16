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
scale_factor = 0.5
# Folder clearing
output_dir = '/Users/joshua.stanley/Desktop/Science Research/Model outputs/R-CNN/zoomed_bounding_boxes/test_dataset/'

stopped = False

response = 'Data will be provided if detections show disease'

apple_classes = ['Apple__black_rot', 'Apple__healthy', 'Apple__rust', 'Apple__scab']
casava_classes = ['Cassava__bacterial_blight', 'Cassava__brown_streak_disease', 'Cassava__green_mottle', 'Cassava__healthy', 'Cassava__mosaic_disease']
cherry_classes = ['Cherry__healthy', 'Cherry__powdery_mildew']
chili_classes = ['Chili__healthy', 'Chili__leaf curl', 'Chili__leaf spot', 'Chili__whitefly', 'Chili__yellowish']
citrus_classes = ['Black spot', 'canker', 'greening', 'healthy']
coffee_classes = ['Coffee__cercospora_leaf_spot', 'Coffee__healthy', 'Coffee__red_spider_mite', 'Coffee__rust']
corn_classes = ['Corn__common_rust', 'Corn__gray_leaf_spot', 'Corn__healthy', 'Corn__northern_leaf_blight']
cucumber_classes = ['Cucumber__diseased', 'Cucumber__healthy']
grape_classes = ['Grape__black_measles', 'Grape__black_rot', 'Grape__healthy', 'Grape__leaf_blight_(isariopsis_leaf_spot)']
guava_classes = ['Gauva__diseased', 'Gauva__healthy']
jamun_classes  = ['Jamun__diseased', 'Jamun__healthy']
lemon_classes  = ['Lemon__diseased', 'Lemon__healthy']
mango_classes = ['Mango__diseased', 'Mango__healthy']
peach_classes = ['Peach__bacterial_spot', 'Peach__healthy']
pepper_classes = ['Pepper_bell__bacterial_spot', 'Pepper_bell__healthy']
pomegranate_classes = ['Pomegranate__diseased', 'Pomegranate__healthy']
potato_classes = ['Potato__early_blight', 'Potato__healthy', 'Potato__late_blight']
rice_classes = ['Rice__brown_spot', 'Rice__healthy', 'Rice__hispa', 'Rice__leaf_blast', 'Rice__neck_blast']
soybean_classes = ['Soybean__bacterial_blight', 'Soybean__caterpillar', 'Soybean__diabrotica_speciosa', 'Soybean__downy_mildew', 'Soybean__healthy', 'Soybean__mosaic_virus', 'Soybean__powdery_mildew', 'Soybean__rust', 'Soybean__southern_blight']
strawberry_classes = ['Strawberry___leaf_scorch', 'Strawberry__healthy']
sugarcane_classes = ['Sugarcane__bacterial_blight', 'Sugarcane__healthy', 'Sugarcane__red_rot', 'Sugarcane__red_stripe', 'Sugarcane__rust']
tea_classes = ['Tea__algal_leaf', 'Tea__anthracnose', 'Tea__bird_eye_spot', 'Tea__brown_blight', 'Tea__healthy']
tomato_classes = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

classes = ['apple',
           'casava',
           'cherry',
           'chili',
           'citrus',
           'coffee',
           'corn',
           'cucumber',
           'grape',
           'guava',
           'jamun',
           'lemon',
           'mango',
           'peach',
           'pepper',
           'pomegranate',
           'potato',
           'rice',
           'soybean',
           'strawberry',
           'sugarcane',
           'tea',
           'tomato'
]
disease_dictionary = {
    "apple": [
        {
            "name": "black_rot",
            "description": "Caused by the fungus Botryosphaeria obtusa, it leads to reddish-brown sunken areas on limbs and brown lesions on fruits. Control involves removing infected limbs and applying copper-based products early in the season8."
        },
        {
            "name": "scab",
            "description": "A fungal disease causing brown spots on leaves and fruits. Prevention includes planting resistant varieties like Honeycrisp and using fungicides containing fenarimol or myclobutanil54."
        },
        {
            "name": "rust (Apple Cedar Rust)",
            "description": "Not detailed in the search results, but generally involves removing infected parts and using fungicides."
        }
    ],
    "cassava": [
        {
            "name": "mosaic_disease",
            "description": "Cassava Mosaic Disease (CMD) is a viral disease spread by whiteflies. Prevention includes using disease-free planting material, resistant varieties, and integrated pest management strategies2."
        },
        {
            "name": "brown_streak_disease",
            "description": "Cassava Brown Streak Disease (CBSD) is characterized by brown streaks on stems. Control involves planting tolerant varieties and removing infected plants6."
        }
    ],
    "cherry": [
        {
            "name": "powdery_mildew",
            "description": "Powdery Mildew is a fungal disease that can cause defoliation. Prevention involves applying protectant fungicides like sulfur products before symptoms appear37."
        }
    ],
    "chili": [
        {
            "name": "leaf curl",
            "description": "Leaf Curl in chili peppers is typically caused by viruses transmitted by whiteflies. Prevention involves using resistant varieties and controlling whitefly populations."
        },
        {
            "name": "leaf spot",
            "description": "Leaf Spot in chili peppers can be caused by fungal or bacterial pathogens and is treated with fungicides or bactericides."
        }
    ],
    "citrus": [
        {
            "name": "black spot",
            "description": "Black Spot in citrus is a fungal disease causing black spots on leaves. It is treated with fungicides."
        },
        {
            "name": "canker",
            "description": "Citrus Canker is a bacterial disease causing lesions on bark. Control involves removing infected bark."
        }
    ],
    "coffee": [
        {
            "name": "cercospora_leaf_spot",
            "description": "Cercospora Leaf Spot in coffee is a fungal disease causing spots on leaves. It is treated with fungicides."
        },
        {
            "name": "rust",
            "description": "Coffee Rust is a fungal disease causing orange spores on leaves. It is treated with fungicides."
        }
    ],
    "corn": [
        {
            "name": "common_rust",
            "description": "Common Rust in corn is a fungal disease causing orange spores on leaves. It is treated with fungicides."
        },
        {
            "name": "gray_leaf_spot",
            "description": "Gray Leaf Spot in corn is a fungal disease causing gray spots on leaves. It is treated with fungicides."
        }
    ],
    "cucumber": [
        {
            "name": "diseased",
            "description": "Cucumber plants affected by general fungal diseases can be treated with fungicides and good hygiene practices."
        }
    ],
    "grape": [
        {
            "name": "black_rot",
            "description": "Black Rot in grapes is a fungal disease causing black lesions on fruits. It is treated with fungicides."
        },
        {
            "name": "black_measles",
            "description": "Black Measles in grapes is a fungal disease causing black spots on leaves. It is treated with fungicides."
        }
    ],
    "guava": [
        {
            "name": "diseased",
            "description": "Guava plants suffering from general fungal diseases can be treated with fungicides and good hygiene practices."
        }
    ],
    "jamun": [
        {
            "name": "diseased",
            "description": "Jamun trees affected by general fungal diseases can be managed with fungicides and good hygiene practices."
        }
    ],
    "lemon": [
        {
            "name": "diseased",
            "description": "Lemon trees affected by general fungal diseases can be treated with fungicides and good hygiene practices."
        }
    ],
    "mango": [
        {
            "name": "diseased",
            "description": "Mango trees affected by general fungal diseases can be managed with fungicides and good hygiene practices."
        }
    ],
    "peach": [
        {
            "name": "bacterial_spot",
            "description": "Peach Bacterial Spot is a bacterial disease causing spots on leaves, treated with bactericides."
        }
    ],
    "pepper (Bell Pepper)": [
        {
            "name": "bacterial_spot",
            "description": "Bell Pepper Bacterial Spot is a bacterial disease causing spots on leaves, treated with bactericides."
        }
    ],
    "pomegranate": [
        {
            "name": "diseased",
            "description": "Pomegranate trees affected by general fungal diseases can be treated with fungicides and good hygiene practices."
        }
    ],
    "potato": [
        {
            "name": "early_blight",
            "description": "Early Blight in potatoes is a fungal disease causing spots on leaves, treated with fungicides."
        },
        {
            "name": "late_blight",
            "description": "Late Blight in potatoes is a fungal disease causing white spores on leaves, treated with fungicides."
        }
    ],
    "rice": [
        {
            "name": "brown_spot",
            "description": "Brown Spot in rice is a fungal disease causing brown spots on leaves, treated with fungicides."
        },
        {
            "name": "leaf_blast",
            "description": "Leaf Blast in rice is a fungal disease causing white spots on leaves, treated with fungicides."
        }
    ],
    "soybean": [
        {
            "name": "bacterial_blight",
            "description": "Soybean Bacterial Blight is a bacterial disease causing spots on leaves, treated with bactericides."
        },
        {
            "name": "powdery_mildew",
            "description": "Powdery Mildew in soybeans is a fungal disease causing white spores on leaves, treated with fungicides."
        }
    ],
    "strawberry": [
        {
            "name": "leaf_scorch",
            "description": "Leaf Scorch in strawberries is a fungal disease causing scorched leaves, treated with fungicides."
        }
    ],
    "sugarcane": [
        {
            "name": "bacterial_blight",
            "description": "Sugarcane Bacterial Blight is a bacterial disease causing spots on leaves, treated with bactericides."
        },
        {
            "name": "red_rot",
            "description": "Red Rot in sugarcane is a fungal disease causing red lesions on stalks, treated with fungicides."
        }
    ],
    "tea": [
        {
            "name": "anthracnose",
            "description": "Anthracnose in tea is a fungal disease causing spots on leaves, treated with fungicides."
        }
    ],
    "tomato": [
        {
            "name": "bacterial_spot",
            "description": "Tomato Bacterial Spot is a bacterial disease causing spots on leaves, treated with bactericides."
        },
        {
            "name": "early_blight",
            "description": "Early Blight in tomatoes is a fungal disease causing spots on leaves, treated with fungicides."
        },
        {
            "name": "late_blight",
            "description": "Late Blight in tomatoes is a fungal disease causing white spores on leaves, treated with fungicides."
        }
    ]
}


found_diseases = []

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("System")
        customtkinter.set_default_color_theme("blue")

        self.title("Sci Research 1.0")
        self.geometry("1200x800")  # Adjusted size for better visibility
        self.minsize(800, 600)  # Set a minimum size

        # Configure grid layout for the main window
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar Frame
        self.sidebar_frame = customtkinter.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6), weight=0)
        self.sidebar_frame.grid_rowconfigure(7, weight=1)  # For radio buttons at the bottom

        # Logo Label
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="SciResearch v1",
            font=customtkinter.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Run Button
        self.sidebar_button_run = customtkinter.CTkButton(
            self.sidebar_frame,
            command=self.run_button_event,
            text='Run',
            width=160
        )
        self.sidebar_button_run.grid(row=1, column=0, padx=20, pady=(10, 5))

        # Stop Button
        self.sidebar_button_stop = customtkinter.CTkButton(
            self.sidebar_frame,
            command=self.stop_button_event,
            text='Stop',
            state='disabled',
            width=160
        )
        self.sidebar_button_stop.grid(row=2, column=0, padx=20, pady=5)

        # File Selection Button
        self.file_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Select File",
            command=self.select_file,
            width=160
        )
        self.file_button.grid(row=3, column=0, padx=20, pady=5)

        # Radio Buttons for Detection Mode
        self.radio_var = tk.IntVar(value=0)
        self.radio_button_live = customtkinter.CTkRadioButton(
            self.sidebar_frame,
            text="Live Detection",
            variable=self.radio_var,
            value=0
        )
        self.radio_button_live.grid(row=4, column=0, padx=20, pady=(20, 5), sticky="w")

        self.radio_button_file = customtkinter.CTkRadioButton(
            self.sidebar_frame,
            text="File Detection",
            variable=self.radio_var,
            value=1
        )
        self.radio_button_file.grid(row=5, column=0, padx=20, pady=5, sticky="w")

        # Spacer to push radio buttons to the bottom
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        # Main Content Frame
        self.main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Video Display
        self.video_label = customtkinter.CTkLabel(self.main_frame, text='', anchor="center")
        self.video_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Lower Frame for Response
        

        

        # Tab View
        self.tabview = customtkinter.CTkTabview(self.main_frame, width=350, height=800)
        self.tabview.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.tabview.add("Output")
        self.tabview.add("Settings")
        self.tabview.grid_rowconfigure(0, weight=1)
        self.tabview.grid_columnconfigure(0, weight=1)

        # Plant Selection Dropdown in Output Tab
        self.plant_dropdown = customtkinter.CTkOptionMenu(
            self.tabview.tab('Output'),
            dynamic_resizing=True,
            values=classes,
            command=self.model_select
        )
        self.plant_dropdown.set("Select Plant")
        self.plant_dropdown.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        # Disease Information Display
        self.disease_info_label = customtkinter.CTkLabel(
            self.tabview.tab('Output'),
            text="Disease info and treatment:",
            anchor="w",
            justify="left",
            font=customtkinter.CTkFont(weight="bold")
        )
        self.disease_info_label.grid(row=1, column=0, padx=20, pady=(10, 5), sticky="w")

        self.disease_info_text = customtkinter.CTkTextbox(
            self.tabview.tab('Output'),
            height=800,
            width=400,
            wrap="word"
        )
        self.disease_info_text.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.disease_info_text.configure(state='disabled')  # Initially disabled

        # Settings Tab - Confidence Slider
        self.confidence_slider = customtkinter.CTkSlider(
            self.tabview.tab('Settings'),
            from_=0,
            to=1,
            number_of_steps=100,
            command=self.confidence_function
        )
        self.confidence_slider.set(confidence)
        self.confidence_slider.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        self.confidence_label = customtkinter.CTkLabel(
            self.tabview.tab('Settings'),
            text=f'Confidence > {confidence * 100:.1f}%',
            anchor="w"
        )
        self.confidence_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")
        
        self.zoom_slider = customtkinter.CTkSlider(
            self.tabview.tab('Settings'),
            from_=0,
            to=1,
            number_of_steps=100,
            command=self.zoom_function
        )
        self.zoom_slider.set(scale_factor)
        self.zoom_slider.grid(row=1, column=0, padx=20, pady=(20, 10), sticky="ew")
        self.zoom_label = customtkinter.CTkLabel(
            self.tabview.tab('Settings'),
            text='zoom',
            anchor="w"
        )
        self.zoom_label.grid(row=2, column=0, padx=20, pady=(20, 10), sticky="ew")

        # Initialize variables
        self.plant_type = 'apple'
        self.file_path = None

    def get_selected_value(self):
        self.plant_type = self.plant_dropdown.get()
        print(f"Selected Plant: {self.plant_type}")
        self.model_select(self.plant_type)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
        )
        if self.file_path:
            print(f"Selected file: {self.file_path}")

    def run_button_event(self):
        global stopped, scale_factor
        scale_factor = 0.4 
        stopped = False
        self.sidebar_button_run.configure(state="disabled", text="Running...")
        self.sidebar_button_stop.configure(state='enabled', text='Stop')
        if self.radio_var.get() == 0:
            threading.Thread(target=self.live_detection_thread, daemon=True).start()
        else:
            threading.Thread(target=self.file_detection_thread, daemon=True).start()

    def stop_button_event(self):
        global stopped
        stopped = True
        self.sidebar_button_run.configure(state="enabled", text="Run")
        self.sidebar_button_stop.configure(state='disabled', text='Stop')     

    def model_select(self, crop):
        global leaf_model, class_name, input_details, output_details

        # Update self.plant_type based on the selected dropdown value
        self.plant_type = crop

        # Load TFLite model and allocate tensors
        leaf_model = tf.lite.Interpreter(model_path=f'/Users/joshua.stanley/Desktop/Science Research/Saved Models/tflite2/model{crop}.tflite')
        leaf_model.allocate_tensors()

        # Get input and output details
        input_details = leaf_model.get_input_details()
        output_details = leaf_model.get_output_details()

        # Print input/output details for debugging
        print("pInput Details:", input_details)
        print("Output Details:", output_details)

        class_name = globals()[f'{crop}_classes']
        print(f"Selected Crop: {crop}")
        print(f"Class Names: {class_name}")
        print(f"Plant type updated to: {self.plant_type}")

    def confidence_function(self, value):
        global confidence
        confidence = float(value)
        self.confidence_label.configure(text=f'Confidence > {confidence * 100:.1f}%')
        print(f'Confidence > {confidence * 100:.1f}%')
    def zoom_function(self, value):
        global scale_factor
        scale_factor = float(value)
        
    def update_disease_info(self, info):
        self.disease_info_text.configure(state='normal')
        self.disease_info_text.delete("1.0", tk.END)
        self.disease_info_text.insert(tk.END, info)

        self.disease_info_text.configure(state='disabled')

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
              # Adjust this factor as needed (e.g., 0.5 for 50% size)
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

            results = model(frame)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    if stopped:
                        break
                    if conf >= confidence:
                        cropped_leaf = frame[y1:y2, x1:x2]
                        if cropped_leaf.size == 0:
                            print("Cropped leaf is empty! Check bounding box coordinates.")
                        else:
                            print("Cropped leaf shape:", cropped_leaf.shape)

                        cropped_leaf = cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB)
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # Unique timestamp
                        save_path = os.path.join(output_dir, f"leaf_{timestamp}.jpg")
                        cv2.imwrite(save_path, cv2.cvtColor(cropped_leaf, cv2.COLOR_RGB2BGR))
                        print(f"Saved cropped leaf to {save_path}")
                        data = tf.image.resize(cropped_leaf, [128, 128])
                        data = np.expand_dims(data, axis=0)  # Shape [1, 128, 128, 3], assuming model expects this

                        # Set the input tensor
                        leaf_model.set_tensor(input_details[0]['index'], data)

                        # Invoke the model
                        leaf_model.invoke()

                        # Get the prediction output
                        prediction = leaf_model.get_tensor(output_details[0]['index'])  # This will give you the raw output array

                        # Fixed preprocessing pipeline
                        confidence_threshold = 0.4  # Set your desired confidence threshold

                        predicted_class = np.argmax(prediction[0])
                        conf_model = prediction[0][predicted_class]  # Get confidence for the predicted class

                        if conf_model < confidence_threshold:  # Check if confidence is below the threshold
                            class_label = "get closer to leaf"
                            color = (255, 255, 0)  # Yellow for low confidence
                            label = class_label
                        else:
                            class_label = class_name[predicted_class]
                            color = (0, 255, 0) if 'healthy' in class_label else (0, 0, 255)
                            label = f"{class_label} ({conf_model:.2f})"

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
        self.sidebar_button_run.configure(state="enabled", text="Run")

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
            
            
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

            results = model(frame)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    if stopped:
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                    if conf >= confidence:
                        cropped_leaf = frame[y1:y2, x1:x2]
                        if cropped_leaf.size == 0:
                            print("Cropped leaf is empty! Check bounding box coordinates.")
                        else:
                            print("Cropped leaf shape:", cropped_leaf.shape)
                        cropped_leaf = cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB)
                        data = tf.image.resize(cropped_leaf, [128, 128])
                        data = np.expand_dims(data, axis=0)  # Shape [1, 128, 128, 3], assuming model expects this

                        # Set the input tensor
                        leaf_model.set_tensor(input_details[0]['index'], data)

                        # Invoke the model
                        leaf_model.invoke()

                        # Get the prediction output
                        prediction = leaf_model.get_tensor(output_details[0]['index'])  # This will give you the raw output array

                        # Fixed preprocessing pipeline
                        confidence_threshold = 0.4  # Set your desired confidence threshold

                        predicted_class = np.argmax(prediction[0])
                        conf_model = prediction[0][predicted_class]  # Get confidence for the predicted class

                        if conf_model < confidence_threshold:  # Check if confidence is below the threshold
                            class_label = "get closer to leaf"
                            color = (255, 255, 0)  # Yellow for low confidence
                            label = class_label
                        else:
                            class_label = class_name[predicted_class]
                            color = (0, 255, 0) if 'healthy' in class_label else (0, 0, 255)
                            label = f"{class_label} ({conf_model:.2f})"

                        print(class_label)

                        try:
                            disease_info = disease_dictionary[self.plant_type][predicted_class]
                            if disease_info['description'] not in found_diseases:
                                found_diseases.append(disease_info['description'])
                                found_diseases.append('\n')
                                self.update_disease_info(found_diseases)
                        except:
                            pass

                        # Draw rectangle and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
            self.video_label.configure(image=img)
            self.video_label.image = img
        cap.release()
        cv2.destroyAllWindows()
        self.sidebar_button_run.configure(state="enabled", text="Run")    

if __name__ == "__main__":
    app = App()
    app.mainloop()