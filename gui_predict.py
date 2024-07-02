

import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

class CropHealthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Crop Health Checker")
        self.root.geometry("800x600")

        self.model = tf.keras.models.load_model('crop_health_model.h5')
        self.class_indices = {0: 'Blight', 1: 'Common_Rust', 2: 'Gray_Leaf_Spot', 3: 'Healthy'}
        self.disease_info = {
            'Blight': "BLIGHTS IMAGE: Environmental conditions that promote the disease are moderate temperatures (18°C - 27°C), moist conditions and long dew periods. The fungus survives on maize leaf residues, and multiple secondary infections develop throughout the season from existing lesions.The lesions are usually noticed on the bottom leaves first, and as spores are released under favorable conditions from these lesions, the upper leaves are infected and it seems like the disease creeps up the plant.\n\nTREATMENT\nFungicides, hybrid selections, crop rotation and ploughing in of plant residues",
            'Common_Rust': "COMMON RUST: Environmental conditions that promote the disease are moderate temperatures (16°C - 25°C) and moist conditions (>95% humidity). Common rust infection is promoted by dew/fog conditions – especially during the night when spores on the leaf surface germinate and penetrate the leaf through the stomata.\nEarly signs of rust infection are visible approximately five days after initial infection as small flecks on the leaves, and proper rust pimples (ten to 14 days) then develop and release spores. On release these spores can be distributed across great distances by the wind. These spores can even infect the same plant again within the same season.\n\nTREATMENT\nFungicides and hybrid selection",
            'Gray_Leaf_Spot': "GRAY LEAF SPOTS: Environmental conditions that promote the disease are moderate to high temperatures and high humidity (>95%).\n\nThe disease symptoms become particularly visible around flowering – although they can be visible earlier under high disease pressure conditions. This is a leaf disease that occurs specifically in KwaZulu-Natal and the eastern Highveld, where fog and/or high humidity and high temperatures prevail.\nA misperception exists that the spores of the fungus are only airborne and distributed by the wind and the rain. Spores can indeed be spread by the wind, but these are secondary spores that can be spread from one planting to the next later in the season – and usually over shorter distances.\n\nTREATMENT\nFungicides, hybrid selections, crop rotation and ploughing in of plant residues",
            'Healthy': "The crop is healthy. No treatment needed."
        }

        self.label = Label(root, text="Upload a crop image to check its health")
        self.label.pack()

        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack()

        self.result_label = Label(root, text="", wraplength=700, justify="left")
        self.result_label.pack()

        self.image_label = Label(root)
        self.image_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img = img.resize((224, 224), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

            predicted_class, confidence = self.predict_image(file_path)
            disease_info_text = self.disease_info.get(predicted_class, "Information not available")
            self.result_label.config(text=f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}\n\n{disease_info_text}')

    def predict_image(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = self.model.predict(img_array)
        class_idx = np.argmax(predictions)
        predicted_class = self.class_indices[class_idx]
        confidence = np.max(predictions)
        
        return predicted_class, confidence

if __name__ == "__main__":
    root = tk.Tk()
    app = CropHealthApp(root)
    root.mainloop()
