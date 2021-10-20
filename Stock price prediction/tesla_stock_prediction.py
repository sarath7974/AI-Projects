#Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential   
from keras.layers import LSTM            
from keras.layers import Dense           
from keras.layers import Dropout   
from keras.callbacks import EarlyStopping

#Loading Dataset
dataset = pd.read_csv("/home/syam/Desktop/sarath/AI  projects/stock market/TSLA.csv")

#Splitting Dataset into Train and Test set
split_row = len(dataset)- int(0.2 * len(dataset))
train_data = dataset.iloc[:split_row]
test_data = dataset.iloc[split_row:]
print("SHAPE OF TRAIN DATA :",train_data.shape)
print("SHAPE OF TEST DATA :",test_data.shape)
train_data.head()

#Load the closing value of stock
train_dataset = train_data.loc[:,["Close"]].values
train_dataset.shape

#Visualise Close value of Dataset
plt.figure(figsize=(16,8))
plt.plot(dataset["Close"])
plt.title("Tesla Stock Closing Price")
plt.show()

#Scaling Data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train_dataset)

#Creating data with Timesteps
def timesteps(train_data):
    time_steps= 60
    X_train = []
    y_train = []
    for i in range(60,len(train_data)):
        # append value of (i-60 :i,0) to X_train , take 60 values of the coloumn index 0 and append to the array
        X_train.append(train_data[i-time_steps : i,0])
        # append value of (i-60 :i) to y_train
        y_train.append(train_data[i,0])
    #convert to array
    X_train,y_train =np.array(X_train),np.array(y_train)     
    # reshape the array to make it 3D by adding 1 as feature
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train,y_train

X_train,y_train = timesteps(train_scaled) 
print("size of X_train",X_train.shape)
print("size of y_train",y_train.shape)

#Build LSTM Model
regressor=Sequential()
"""return_sequences=True : to return  the full sequence output
   input_shape: is the shape of training set
   units :is the dimension of cell in LSTM"""
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (60,1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))
regressor.summary()

#Compile the model and fit the model
callback = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
regressor.compile(optimizer="adam",loss="mean_squared_error")
model=regressor.fit(X_train,y_train,epochs=50,batch_size=32,validation_split=0.1,verbose=1, callbacks=[callback])

#Visualising Training and Validation Loss
loss = model.history['loss']
val_loss = model.history['val_loss']
epochs = range(len(loss))
plt.figure(figsize=(10,6))
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

#Testing the Model
#Merge train and test data on axis=0
#we need the 60 days’ price before the 1st date in the test dataset
dataset_total=(pd.concat((train_data["Close"],test_data["Close"]),axis=0))
model_input = dataset_total[len(dataset_total) - len(test_data) - 60:].values

#Reshape & Scale the Training Data
model_input=model_input.reshape(-1,1)   #reshape
model_input=scaler.fit_transform(model_input)   #scale
model_input
print("SHAPE OF MODELINPUT :",model_input.shape)

#Creating data with Timesteps
time_steps= 60
X_test = []
for i in range(60,len(model_input)):
    X_test.append(model_input[i-time_steps : i,0])
    
X_test = np.array(X_test)     
# reshape the array to make it 3D by adding 1 as feature
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print("shape of X_test ",X_test.shape)

#actual price of stock
actual_price=test_data.loc[:,["Close"]].values

#Predicting Stock Price using Test Data
predicted_price = regressor.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)

#Visualise Actual vs Predicted Stock Price
plt.figure(figsize=(16,8))
plt.plot(actual_price,color="red",label="actual_price")
plt.plot(predicted_price,color=("green"),label="predicted_price")
plt.title("Actual Price vs Predicted Stock Price")
plt.xlabel("time")
plt.ylabel("price")
plt.legend()
plt.show()