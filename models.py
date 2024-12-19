# MPI imports
from mpi4py import MPI
import sys
# Models imports
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

# To show which process is sunning we used color changer
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

def color(rank):
    if rank == 0:
        return RED
    elif rank == 1:
        return GREEN
    elif rank == 2:
        return YELLOW

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f'I am process {rank} which is the master \n')
# Read data from file as a dataframe
data = pd.read_csv('/home/adnane/mpi/a-project-of-supervised-machine-learning-using-gradient-boosting/data.csv', encoding='latin-1') # I used encoding because it could not run directly

# Data cleaning for model training
# Drop unwanted columns
# we will not use all Columns because it exist unecessary ones  
# 
data = data[
    ['Order Date','Ship Date','Ship Mode','Segment','Country','Market','Region','Category','Sub-Category','Sales','Quantity','Profit','Shipping Cost','Order Priority'		
]]

# print(data.head(10))
print("Unwanted column has been dropped succesfully !\n")

# Remove unwanted rows

# We will not use all rows  because we are interseted only in african market 


data = data[data.isin(['Africa']).any(axis=1)]
# print(data.head(10))
# print("number of rows is : ",data.count())
# print('\n\n\n')
# print(data.dtypes)
print("Unwanted rows has been deleted succesfully !\n")


# Replace dates of `order` and `shipping` with `shiping days` for simplicity. 


data['Order Date'] = pd.to_datetime(data['Order Date'],format="%d-%m-%Y")
data['Ship Date'] = pd.to_datetime(data['Ship Date'],format="%d-%m-%Y")

# Calculate difference
data['Shipping days'] = data['Ship Date'] - data['Order Date'] 
data['Shipping days'] = data['Shipping days'].astype('str')
data['Shipping days'] = data['Shipping days'].str.replace(' days','')
data['Shipping days'] = data['Shipping days'].astype('int64')


data = data.drop(columns=['Order Date', 'Ship Date'])

# print(data.dtypes)
print("shiping  has been changed to int64 succesfully !\n")

# EDA
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
# Send/Broadcast data to all other process 
preprocessor = comm.bcast(preprocessor, root=0)
X_train = comm.bcast(X_train, root=0)
X_test = comm.bcast(X_test, root=0)
y_train = comm.bcast(y_train, root=0)
y_test = comm.bcast(y_test, root=0)
comm.Barrier()
 
# regressors list
regressors = [
    GradientBoostingRegressor(random_state=42), # Yellow 
    LinearRegression(), # Green
    KNeighborsRegressor() # Red
]

# Pipeline for model
pipeline = make_pipeline(preprocessor,regressors[rank],verbose=True) if rank < len(regressors) else None

MSE = [0]*3
R2 = [0] *3
MAE = [0]*3
RMSE = [0]*3

if pipeline is not None:
    # Training
    print(f'{color(rank)}Fit the model{RESET}')
    pipeline.fit(X_train, y_train)
    comm.Barrier()
    print(f"{color(rank)} {type(regressors[rank]).__name__} trained.")
    y_pred = pipeline.predict(X_test)
    
    # Generate Regressor report
    MSE[rank] = mean_squared_error(y_test, y_pred) # measurement of the typical absolute discrepancies between a dataset's actual values and projected values.
    # comm.Barrier() # Ensure all processes wait for each other
    R2[rank] = r2_score(y_test, y_pred) # measures the square root of the average discrepancies between a dataset's actual values and projected values
    # comm.Barrier() # Ensure all processes wait for each other
    MAE[rank] = mean_absolute_error(y_test, y_pred) # A statistical metric frequently used to assess the goodness of fit of a regression model is the R-squared (R2) score,
    # comm.Barrier() # Ensure all processes wait for each other
    RMSE[rank] = mean_squared_error(y_test, y_pred)

    # type(regressors[rank]).__name__ Shows model name
    print(f"{color(rank)}[{rank} process]: {type(regressors[rank]).__name__}")
    print(f'{color(rank)}MAE for {type(regressors[rank]).__name__} : ',MAE[rank])
    print(f'{color(rank)}MSE {type(regressors[rank]).__name__} : ',MSE[rank])
    print(f'{color(rank)}R2  {type(regressors[rank]).__name__}: ',R2[rank])
    print(f'{color(rank)}RMSE {type(regressors[rank]).__name__} : ',RMSE[rank])

    # Visualization of the prediction error .
    disp = PredictionErrorDisplay.from_estimator(pipeline, X_test, y_test)
    plt.savefig(f'Prediction error of {type(regressors[rank]).__name__}')
