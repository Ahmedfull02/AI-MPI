# MPI imports
from mpi4py import MPI

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


# Preprcessing, pipeline, metrics libraries imports


from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_error, PredictionErrorDisplay


# data, plots libraries imports


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# use 1st process
if rank == 0:
    # Read data from file as a dataframe
    data = pd.read_csv('/home/adnane/mpi/a-project-of-supervised-machine-learning-using-gradient-boosting/data.csv', encoding='latin-1') # I used encoding because it could not run directly

    print(data.head(10))
    print("number of rows is : ",data.count())


    # Data cleaning for model training

    # Drop unwanted columns
    # we will not use all Columns because it exist unecessary ones  
    # 


    data = data[
        ['Order Date','Ship Date','Ship Mode','Segment','Country','Market','Region','Category','Sub-Category','Sales','Quantity','Profit','Shipping Cost','Order Priority'		
    ]]

    print(data.head(10))


    # Remove unwanted rows

    # We will not use all rows  because we are interseted only in african market 


    data = data[data.isin(['Africa']).any(axis=1)]
    print(data.head(10))
    print("number of rows is : ",data.count())
    print('\n\n\n')
    print(data.dtypes)


    # Replace dates of `order` and `shipping` with `shiping days` for simplicity. 


    data['Order Date'] = pd.to_datetime(data['Order Date'],format="%d-%m-%Y")
    data['Ship Date'] = pd.to_datetime(data['Ship Date'],format="%d-%m-%Y")

    # Calculate difference
    data['Shipping days'] = data['Ship Date'] - data['Order Date'] 
    data['Shipping days'] = data['Shipping days'].astype('str')
    data['Shipping days'] = data['Shipping days'].str.replace(' days','')
    data['Shipping days'] = data['Shipping days'].astype('int64')


    data = data.drop(columns=['Order Date', 'Ship Date'])

    print(data.dtypes)


    # ## EDA
    #  Understanding data

    # Count the occurrences of each order priority 1st process

    order_count = data['Order Priority'].value_counts()
    # Create the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(order_count, labels=order_count.index,autopct='%1.1f%%', colors=['blue', 'yellow', 'red', 'green'])
    plt.title('Order Priority Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Show the plot
    plt.savefig('Pie chart')
    
    # Group the data by 'Sub-Category' and calculate the sum of 'Profit' for each sub-category
    profit = data.groupby('Sub-Category')['Profit'].sum()
    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    profit.plot(kind='bar', color='skyblue')

    plt.title('Total Profit for Each Sub-Category', fontsize=14)
    plt.xlabel('Sub-Category', fontsize=12)
    plt.ylabel('Total Profit', fontsize=12)

    plt.savefig('Bar chart')

# Splitting data
# choose dependant vars and undependant var.
X = data.drop(['Shipping Cost'],axis=1)
y = data['Shipping Cost']

# Split data into Train and validation set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Define categorical and numerical features


categorical_features = X.select_dtypes(
   include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
   include=["float64", "int64"]
).columns.tolist()


# Preprocessing


preprocessor = ColumnTransformer(
   transformers=[
       ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
       ("num", StandardScaler(), numerical_features),
   ]
)


# # First Model: `Gradient Boosting Regressor` 

# Pipeline for model
pipeline1 = make_pipeline(preprocessor,
                         GradientBoostingRegressor(random_state=42),
                         verbose=True   
                   )


## Access  parameters of a GBRegressor by default


classifier_params = pipeline1['gradientboostingregressor'].get_params()
print("\nClassifier parameters:")
for param, value in classifier_params.items():
    print(f"{param}: {value}")


# Training


print('Fit the model')
pipeline1.fit(X_train, y_train)
y_pred = pipeline1.predict(X_test)

# Generate Gradient Boosting Regressor report
MSE = mean_squared_error(y_test, y_pred) # measurement of the typical absolute discrepancies between a dataset's actual values and projected values.
R2 = r2_score(y_test, y_pred) # measures the square root of the average discrepancies between a dataset's actual values and projected values
MAE = mean_absolute_error(y_test, y_pred) # A statistical metric frequently used to assess the goodness of fit of a regression model is the R-squared (R2) score,
RMSE = mean_squared_error(y_test, y_pred)

print("Gradient Boosting Regressor Report:")
print('MAE : ',MAE)
print('MSE : ',MSE)
print('R2 : ',R2)
print('RMSE : ',RMSE)


## Visualization of the prediction error of a GB model.


disp = PredictionErrorDisplay.from_estimator(pipeline1, X, y)
plt.savefig('Prediction error of GB')


# # Second Model: Linear Regression Regressor

# Pipeline


pipeline2 = make_pipeline(preprocessor,
                         LinearRegression(),
                         verbose=True   
                   )


# Training


pipeline2.fit(X_train, y_train)
y_pred = pipeline2.predict(X_test)

# Generate Linear Regression report
MSE = mean_squared_error(y_test, y_pred) # measurement of the typical absolute discrepancies between a dataset's actual values and projected values.
R2 = r2_score(y_test, y_pred) # measures the square root of the average discrepancies between a dataset's actual values and projected values
MAE = mean_absolute_error(y_test, y_pred) # A statistical metric frequently used to assess the goodness of fit of a regression model is the R-squared (R2) score,
RMSE = mean_squared_error(y_test, y_pred)

print("Linear Regression Report:")
print('MAE : ',MAE)
print('MSE : ',MSE)
print('R2 : ',R2)
print('RMSE : ',RMSE)


## Visualization of the prediction error of a GB model.


disp = PredictionErrorDisplay.from_estimator(pipeline2, X, y)
plt.savefig('Prediction error of LR')


# Absence of `hyperparams`:
# Linear regressor does not have hyper parameters to modify, for that reason it used only by default params only. 

# # Third Model: `KNN regressor`:

# Pipeline for model


pipeline3 = make_pipeline(preprocessor,
                         KNeighborsRegressor(),
                         verbose=True   
                   )


## Access  parameters of a GBRegressor by default


classifier_params = pipeline3['kneighborsregressor'].get_params()
print("\nClassifier parameters:")
for param, value in classifier_params.items():
    print(f"{param}: {value}")


# Training


print('Fit the model')
pipeline3.fit(X_train, y_train)
y_pred = pipeline3.predict(X_test)

# Generate KNN Regression report
MSE = mean_squared_error(y_test, y_pred) # measurement of the typical absolute discrepancies between a dataset's actual values and projected values.
R2 = r2_score(y_test, y_pred) # measures the square root of the average discrepancies between a dataset's actual values and projected values
MAE = mean_absolute_error(y_test, y_pred) # A statistical metric frequently used to assess the goodness of fit of a regression model is the R-squared (R2) score,
RMSE = mean_squared_error(y_test, y_pred)

print("KNNeighbors Regression Report:")
print('MAE : ',MAE)
print('MSE : ',MSE)
print('R2 : ',R2)
print('RMSE : ',RMSE)


## Visualization of the prediction error of a GB model.


disp = PredictionErrorDisplay.from_estimator(pipeline3, X, y)
plt.savefig('Prediction error of KNN')
