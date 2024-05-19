import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

train_dir = './dataset/train'
validation_dir = './dataset/validation'
test_img_path = './images/laddu2.jpg'

model = load_model('dog_breed_classifier_model.h5')

train_datagen = image.ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
class_indices_to_breed_train = {v: k for k, v in train_generator.class_indices.items()}

validation_datagen = image.ImageDataGenerator()
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
class_indices_to_breed_validation = {v: k for k, v in validation_generator.class_indices.items()}

def predict_breed(img_path, class_indices_to_breed):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_breed = class_indices_to_breed[predicted_class_index]

    return predicted_breed, prediction[0][predicted_class_index]

predicted_breed, accuracy = predict_breed(test_img_path, class_indices_to_breed_validation)
simplified_breed_name = " ".join([part.capitalize() for part in predicted_breed.split("-")])
print("Image Selected: ", test_img_path)
print("Predicted breed:", simplified_breed_name)
print("Accuracy:", accuracy)
