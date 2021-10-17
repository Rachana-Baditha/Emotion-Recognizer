import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def initCnnModel(inputshape):

    finaloutputlen = 3

    #Initializing Sequential model
    model = keras.Sequential()
    
    #-----Round 1 Layers-----

    #Convolution Layer
    model.add( layers.Conv2D(64 , (3,3) , padding = 'same' , input_shape = inputshape , activation = 'relu' ) )
    #Max Pooling Layer
    model.add( layers.MaxPooling2D( (2,2) ) )

    #-----Round 2 Layers------

    model.add( layers.Conv2D( 128 , (3,3) , padding = 'same' , activation = 'relu') )
    model.add( layers.MaxPooling2D( (2,2) ) )

    #-----Round 3 Layers-----

    model.add( layers.Conv2D( 512 , (3,3) , padding = 'same' , activation = 'relu' ) )
    model.add( layers.MaxPooling2D( (2,2) ) )

    #Flattening Layer
    model.add( layers.Flatten())

    #Dense Layer
    model.add( layers.Dense(216 , activation= 'relu' ) )

    #Final Dense Layer
    model.add( layers.Dense(finaloutputlen, activation = 'softmax') )

    #Compiling Model
    model.compile( optimizer = 'Adam' , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'] )

    return model

def trainModel(model, xtrain, ytrain ):
    
    #Fitting test data into the model 
    model.fit( xtrain , ytrain , epochs = '16')

def evaluateModel( model, xtest, ytest):

    #Evaluating model performance with test data
    model.evaluate(xtest, ytest)

size = (26,26,3)

newmodel = initCnnModel(size)


    