# predict.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import preprocess  # Import the preprocessing module

def predict_image(image_path, model, class_indices):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    class_names = list(class_indices.keys())
    
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    return predicted_class, confidence

if __name__ == "__main__":
    model = tf.keras.models.load_model('crop_health_model.h5')
    train_dir = './data'  # Replace with the path to your DATA directory
    train_generator, _ = preprocess.get_data_generators(train_dir)
    class_indices = train_generator.class_indices
    
    image_path = 'path/to/your/image.jpg'  # Replace with the path to the image you want to predict
    predicted_class, confidence = predict_image(image_path, model, class_indices)
    print(f'Predicted Class: {predicted_class}, Confidence: {confidence:.2f}')
