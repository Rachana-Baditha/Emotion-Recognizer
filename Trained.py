# import required packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all train images
train_generator = train_data_gen.flow_from_directory('C:\\Files\\Internship\\datasets\\train',target_size=(48, 48),batch_size=64,color_mode="grayscale",class_mode='categorical')

# Preprocess all test images
validation_generator = validation_data_gen.flow_from_directory('C:\\Files\\Internship\\datasets\\test',target_size=(48, 48),batch_size=64,color_mode="grayscale", class_mode='categorical')

# create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

emotion_model.summary()

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Train the neural network/model

emotion_model_info = emotion_model.fit(train_generator, steps_per_epoch= (28709//64) ,epochs=50,validation_data=validation_generator, validation_steps= (7178//64) )

print(emotion_model_info)

# serialize model to json
json_model = emotion_model.to_json()

#save the model architecture to JSON file
with open('emotionrec_model.json', 'w') as json_file:
    json_file.write(json_model)

#saving the weights of the model
emotion_model.save_weights('emotionsrec_weights.h5')
