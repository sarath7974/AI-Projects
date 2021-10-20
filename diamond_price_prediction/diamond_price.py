#Import Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


#Reading Dataset
data = pd.read_csv('/home/syam/Desktop/sarath/AI  projects/diamond/diamond_data/diamonds.csv')
print(data.shape)
data.head()

print(data.info())
print(data.nunique())

# Data Preprocessing
# Remove 'Unnamed:0' column
data = data.drop(["Unnamed: 0"],axis=1)

# convert price value to float
data["price"]= data.price.astype(float)

#Remove features x, y, z containing Zero values
data.loc[(data['x'] == 0) | (data['y'] == 0) | (data['z'] == 0)]
data = data[(data[['x', 'y', 'z']] !=0).all(axis=1)]
print(data.shape)
data.head()

#Feature Engineering
data["size"] = data["x"]* data["y"]* data["z"]
data = data.drop(data[['x', 'y', 'z']], axis=1)
data_1 =data.copy()

#Label Encoder convert categorical data into numerical data
columns = (data.dtypes =="object")
object_col = list(columns[columns].index)
def label_encoding(d):
    """
    Input  : Input dataframe
    _________________________________
    Output : Encoded dataframe
    __________________________________
    Descriptiion : convert categorical features to numerical value
    """
    label_encoder = LabelEncoder()
    for col in object_col:
        d[col] = label_encoder.fit_transform(d[col])
    return d

data=label_encoding(data)


#Visualizing distribution of each features
def visualise(object_col,data):
    """
    Inputs :
    object_col: feature names
    data  : encoded dataframe
    _____________________________________________________
    Description :plot relation between price,count of diamond vs features of the diamonds(cut,color,clarity)
    _____________________________________________________
    """
    fig_1 = plt.figure(figsize = (20, 6), facecolor='#fbe7dd')
    for i in range(len(object_col)):
        fig_1.add_subplot(1, 3, i+1)
        sns.countplot(data_1[object_col[i]], palette='icefire_r')
        plt.title("Count vs %s"%object_col[i])
    plt.show()

    fig_2 = plt.figure(figsize = (20, 6), facecolor='#fbe7dd')
    for i in range(len(object_col)):
        fig_2.add_subplot(1, 3, i+1)
        sns.barplot(x=object_col[i], y="price", data=data_1, palette='icefire_r')
        plt.title("Price vs %s"%object_col[i])
    plt.show()

plot=visualise(object_col,data_1)

#Corelation matrix
#corelation between different features of diamond
corr=data.corr()
plt.figure(figsize = (10,10))
sns.heatmap(corr,vmin=-1, vmax=1, center=0,square=True, annot = True,cmap="Blues")

#Split dataset into Target and Feature
Y =data["price"]
X = data.drop("price",axis=1)

#Scale the data using MinMaxScaler
scaler =MinMaxScaler()
X = scaler.fit_transform(X)

#Split the dataset into Train_set and Test_set
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#Building the Model
# Use cross-validation method to evaluate the best model for predicting the value
Model = {"Linear_Regression": LinearRegression(),
         "Decision_Tree_Regressor": DecisionTreeRegressor(),
         "K_neighbors_Regression": KNeighborsRegressor(),
         "Random_Forest_Regression": RandomForestRegressor()
        }
for name,model in Model.items():
    print(f'Using model: {name}')
    print("score",cross_val_score(model,X_train,Y_train,cv=5).mean())
    print("="*40)

#Predicting the model
model_1 = RandomForestRegressor()
model_1.fit(X_train,Y_train)
Y_pred = model_1.predict(X_test)

for X,Y in list(zip(X_test, Y_test))[:10]:
    print(f"model predicts {model_1.predict([X])[0]}, real value: {Y}")

#Regression Score
score = r2_score(Y_test,Y_pred)*100
print("Regression Score:",score)
print(f"Root_mean_square_error: {np.sqrt(mean_squared_error(Y_test,Y_pred))}")

#Visualize the prediction
fig= plt.subplots(figsize=(12,8))
p1 = max(max(Y_pred), max(Y_test))
p2 = min(min(Y_pred), min(Y_test))
plt.scatter(Y_test,Y_pred)
plt.plot([p1, p2], [p1, p2], "red")
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual_price vs Predicted_price")
plt.show()
