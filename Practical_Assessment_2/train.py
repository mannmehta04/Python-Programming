import os
from data_preprocessing import preprocess_data
from model import create_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

def lr_scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.0001

train_dir = './dataset/train'
validation_dir = './dataset/validation'
image_size = (150, 150)
batch_size = 32
num_epochs = 50

train_generator, validation_generator = preprocess_data(train_dir, validation_dir, image_size, batch_size)
input_shape = (image_size[0], image_size[1], 3)
num_classes = len(os.listdir(train_dir))

model = create_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("dog_breed_classifier_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1)
lr_schedule = LearningRateScheduler(lr_scheduler)

history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stopping, reduce_lr, lr_schedule]
)
