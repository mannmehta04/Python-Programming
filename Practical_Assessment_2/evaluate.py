import os
from data_preprocessing import preprocess_data
from model import create_model

train_dir = './dataset/train'
validation_dir = './dataset/validation'
image_size = (150, 150)
batch_size = 32
num_epochs = 10 

train_generator, validation_generator = preprocess_data(train_dir, validation_dir, image_size, batch_size)
input_shape = (image_size[0], image_size[1], 3)
num_classes = len(os.listdir(train_dir))

model = create_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights('dog_breed_classifier_model.h5')

evaluation = model.evaluate(validation_generator)
print("Loss: ", evaluation[0])
print("Accuracy: ", evaluation[1])
