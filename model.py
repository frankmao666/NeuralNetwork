import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from ann import NeuralNetwork 


data_file = "mammographic_masses.data.txt"
column_names = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]
df = pd.read_csv(data_file, na_values=['?'], names = column_names) # some cell in the data file is missing, convert the missing data to NaN

df.dropna(inplace=True) # drop all nah rows

feature_names = ["Age", "Shape", "Margin", "Density"]
inputs = df[feature_names].values
outputs = df['Severity'].values


# scale the input data
scaler = preprocessing.StandardScaler()
inputs_scaled = scaler.fit_transform(inputs)
#print(inputs_scaled)

######## Create model by tensorflow
def create_model():
    model = Sequential()
    # feature inputs going into an 6-unit layer (more does not seem to help - in fact you can go down to 4)
    model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
    # "Deep learning" turns out to be unnecessary - this additional hidden layer doesn't help either.
    #model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    # Output layer with a binary classification (benign or malignant)
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model; rmsprop seemed to work best
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Wrap our Keras model in an estimator compatible with scikit_learn, have tried higher epochs but did not increase model score
estimator = KerasClassifier(build_fn=create_model, epochs=100, verbose=0)
#use scikit_learn's cross_val_score to evaluate this model identically to the others
cv_scores = cross_val_score(estimator, inputs_scaled, outputs, cv=10) 
print("accuracy of model by Tensorflow:", cv_scores.mean())

######## Create model by my library
layer_dims = [4, 6, 1] # same layers and unit numbers
actFuns = ["ReLU", "sigmoid"] # same activation functions
learning_rate = 0.05
epochs = 10000 # higher epochs contribute to lower cost

k = 10 # the number we want to do cross-validation
total_accuracy = 0
for _ in range(k):

    trainX, testX, trainY, testY = train_test_split(inputs_scaled, outputs, test_size=0.2) # create train and test sets randomly

    trainX = trainX.T # personal library should have (n, m) matrix to be input, n is number of attributes, m is number of samples
    testX = testX.T
    trainY = trainY.reshape(len(trainY), 1).T
    testY = testY.reshape(len(testY), 1).T

    myNeuralNetwork = NeuralNetwork(trainX, trainY, layer_dims, actFuns, learning_rate, epochs) # init the network
    myNeuralNetwork.train() 

    # visualize the cost trend 
    #print(myNeuralNetwork.costs[:10])
    #print(myNeuralNetwork.costs[9990:])

    predictY = myNeuralNetwork.predict(testX) # prediction from the network
    Result = predictY == testY # check if prediction equals to the expected

    accuracy = np.count_nonzero(Result == True) / testY.shape[1] # the # of correct result(True) / total testing # 
    total_accuracy += accuracy
    
print("accuracy of my model:", total_accuracy / k) 