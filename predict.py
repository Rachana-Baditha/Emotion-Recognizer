import numpy as np
from keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np

def load_my_model():
    with open('emotionrec_model.json', 'r') as json_file:
        modelj = json_file.read()

    #load the model architecture 
    emotion_model = model_from_json(modelj)
    #emotion_model.summary()

    emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    return emotion_model

def getPrediction(emotion_model, imgpath):

    emotions = ["Angry","Disgusted","Scared","Happy", "Neutral", "Sad", "Surprised"]

    img_width, img_height = 48,48
    img = image.load_img(imgpath, target_size = (img_width, img_height), grayscale=True)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)

    predjib = emotion_model.predict(img)

    emotenum = np.argmax(predjib, axis = 1)
    return emotions[emotenum[0]]