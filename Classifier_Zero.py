# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:11:24 2020

@author: JackieYan
"""

import datetime
starttime = datetime.datetime.now()

import os
import random
import numpy as np
from tensorflow import set_random_seed

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from sklearn.metrics import classification_report

import efficientnet.keras as efn 
#define seed
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    set_random_seed(0)
seed = 2020
seed_everything(seed)

#Model parameters
batch_size = 32
size = 224
chinnels=3
num_classes = 2
warmup_epochs=5
epochs=30


#load data and augmentation
train_data = '../input/train_dir/'
test_data='../input/test_dir/'

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range = 360,
    zoom_range = 0.1,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.5, 1.5],
    validation_split=0.15,
    #We divide 0.15 of the training set into a validation set
    #preprocessing_function=augment,
  
    rescale=1./255
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=(size,size),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=seed
    )

val_generator= train_datagen.flow_from_directory(
    train_data,
    target_size=(size,size),
    batch_size=1,
    color_mode='rgb',
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=seed
    )

test_generator = test_datagen.flow_from_directory(
    directory=test_data,
    target_size=(size,size),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=seed
)

#label_map = (train_generator.class_indices)



# build modal and load in EfficientNetB0
effnet = efn.EfficientNetB0(weights='imagenet',
                        include_top=False,
                        input_shape=(size, size, chinnels))
def build_model():

    model = Sequential()
    model.add(effnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    return model
model = build_model()        

metric_list = ["accuracy"]
optimizer = Adam(lr=1e-4)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metric_list)

#warm_up
history_warmup = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=warmup_epochs,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    verbose=2  # If occur error that 'Timeout waiting for IOPub output', set verbose to 0.
)    


#fine-tuning CNN
for layer in model.layers:
    layer.trainable = True
    
optimizer =Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=metric_list)

LOG_DIR = './classifier_weights'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
else:
    pass
CKPT_PATH ="./classifier_weights/classifier_zero.h5"

# Create checkpoint callback
checkpoint = ModelCheckpoint(filepath=CKPT_PATH,
                             monitor='val_acc',
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             verbose=1)

reduceLROnPlateau = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    min_lr=0.000001,
    verbose=1,
    mode='min'
)
earlyStopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    mode='min'
)

history_fintune = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[checkpoint, reduceLROnPlateau, earlyStopping],
    verbose=2  # If occur error that 'Timeout waiting for IOPub output', set verbose to 0.
)
        
# report results

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
print(classification_report(test_generator.classes,
	pred.argmax(axis=1), target_names=['NBI','M-NBI']))

endtime = datetime.datetime.now()
print (endtime - starttime)
