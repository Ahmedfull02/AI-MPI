from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_error, PredictionErrorDisplay

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from file as a dataframe

#Read data 
data = pd.read_csv("data.csv", encoding='latin-1') # I used encoding because it could not run directly

print(data.head(10))
print("number of rows is : ",data.count())

# Data cleaning for model training

# Drop unwanted columns
# we will not use all Columns because it exist unecessary ones  

data = data[
    ['Order Date','Ship Date','Ship Mode','Segment','Country','Market','Region','Category','Sub-Category','Sales','Quantity','Profit','Shipping Cost','Order Priority'		
]]

print(data.head(10))

# Remove unwanted rows
# we will not use all rows  because we are interseted only in african market 

data = data[data.isin(['Africa']).any(axis=1)]
print(data.head(10))
print("number of rows is : ",data.count())
print('\n\n\n')
print(data.dtypes)

import pandas as pd

# Convert columns to datetime
data['Order Date'] = pd.to_datetime(data['Order Date'],format="%d-%m-%Y")
data['Ship Date'] = pd.to_datetime(data['Ship Date'],format="%d-%m-%Y")

# Calculate difference
data['Shipping days'] = data['Ship Date'] - data['Order Date'] 
data['Shipping days'] = data['Shipping days'].astype('int64')

data = data.drop(columns=['Order Date', 'Ship Date'])

print(data)


# EDA
# Understanding data
# All columns exist in our data

# Count the occurrences of each order priority
order_count = data['Order Priority'].value_counts()

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(order_count, labels=order_count.index,autopct='%1.1f%%', colors=['blue', 'yellow', 'red', 'green'])
plt.title('Order Priority Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the plot
# plt.show()

# Group the data by 'Sub-Category' and calculate the sum of 'Profit' for each sub-category
profit = data.groupby('Sub-Category')['Profit'].sum()

# Plot the bar chart
plt.figure(figsize=(10, 6))
profit.plot(kind='bar', color='skyblue')

# Add a title and labels
plt.title('Total Profit for Each Sub-Category', fontsize=14)
plt.xlabel('Sub-Category', fontsize=12)
plt.ylabel('Total Profit', fontsize=12)

# Show the plot
plt.tight_layout()
# plt.show()

from collections import Counter
X = data.drop(['Shipping Cost'],axis=1)
y = data['Shipping Cost']

print(X.shape)
print(y.shape)

# Split data into Train and validation set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Make Some **Transformation** into data 

# Define categorical and numerical features
categorical_features = X.select_dtypes(
   include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
   include=["float64", "int64"]
).columns.tolist()

# For numerical data we used **Scaler** and for categorical data it used **OnehotEncoder**
preprocessor = ColumnTransformer(
   transformers=[
       ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
       ("num", StandardScaler(), numerical_features),
   ]
)

# Define the parameter grid for GradientBoostingRegressor
# param_grid_GBooo = {
#     'gradientboostingregressor__n_estimators': [4, 8, 16, 32],
#     'gradientboostingregressor__learning_rate': [1, 0.5, 0.25, 0.1]
# }

"""
First Model Gradient Boosting Regressor
"""
# Parameters for GBRsgressor

param_grid_GB = {
    "gradientboostingregressor__loss":["deviance"],
    "gradientboostingregressor__learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "gradientboostingregressor__min_samples_split": np.linspace(0.1, 0.5, 12),
    "gradientboostingregressor__min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "gradientboostingregressor__max_depth":[3,5,8],
    "gradientboostingregressor__max_features":["log2","sqrt"],
    "gradientboostingregressor__criterion": ["friedman_mse",  "mae"],
    "gradientboostingregressor__subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "gradientboostingregressor__n_estimators":[10]
    }

# *Pipeline* is a series of data processing steps. Instead of creating new instances of each element in our model, we can simply put all those tasks into one pipeline.
pipeline1 = make_pipeline(preprocessor,
                         GradientBoostingRegressor(random_state=42),
                         verbose=True   
                   )

# Fit the model on the training data
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

gsGB = GridSearchCV(estimator=LinearRegression(),
                    param_grid=param_grid_GB,
                    cv=5,
                    scoring='neg_mean_squared_error'
                )
gsGB.fit(X_train, y_train)
best_model_GB = gsGB.best_estimator_

y_pred = best_model_GB.predict(X_test)

# Generate Gradient Boosting Regressor report
MSE = mean_squared_error(y_test, y_pred) # measurement of the typical absolute discrepancies between a dataset's actual values and projected values.
R2 = r2_score(y_test, y_pred) # measures the square root of the average discrepancies between a dataset's actual values and projected values
MAE = mean_absolute_error(y_test, y_pred) # A statistical metric frequently used to assess the goodness of fit of a regression model is the R-squared (R2) score,
RMSE = mean_squared_error(y_test, y_pred)

print("Gradient Boosting Regressor with GSCV Report:")
print('MAE : ',MAE)
print('MSE : ',MSE)
print('R2 : ',R2)
print('RMSE : ',RMSE)

"""
Second Model Linear Regression Regressor
"""
# Linear regression does not have hyperparamters

# *Pipeline* is a series of data processing steps. Instead of creating new instances of each element in our model, we can simply put all those tasks into one pipeline.
pipeline2 = make_pipeline(preprocessor,
                         LinearRegression(),
                         verbose=True   
                   )

# Fit the model on the training data
print('Fit the model')
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

"""
Third Model: KNN Regressor
"""

# Hyper parameter for KNN regressor
parameters_KNN = {
    'n_neighbors': (1,10, 1),
    'leaf_size': (20,40,1),
    'p': (1,2),
    'weights': ('uniform', 'distance'),
    'metric': ('minkowski', 'chebyshev'),
}

# *Pipeline* is a series of data processing steps. Instead of creating new instances of each element in our model, we can simply put all those tasks into one pipeline.
pipeline3 = make_pipeline(preprocessor,
                         KNeighborsRegressor(),
                         verbose=True   
                   )

# Fit the model on the training data
print('Fit the model')
pipeline3.fit(X_train, y_train)
y_pred = pipeline3.predict(X_test)

# Generate Stochastic Gradient Descent Regression report
MSE = mean_squared_error(y_test, y_pred) # measurement of the typical absolute discrepancies between a dataset's actual values and projected values.
R2 = r2_score(y_test, y_pred) # measures the square root of the average discrepancies between a dataset's actual values and projected values
MAE = mean_absolute_error(y_test, y_pred) # A statistical metric frequently used to assess the goodness of fit of a regression model is the R-squared (R2) score,
RMSE = mean_squared_error(y_test, y_pred)

print("KNNeighbors Regression Report:")
print('MAE : ',MAE)
print('MSE : ',MSE)
print('R2 : ',R2)
print('RMSE : ',RMSE)


# cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
# cm_show = ConfusionMatrixDisplay(confusion_matrix=cm)
# cm_show.plot(cmap='Blues')
# plt.title('Confusion matrix')
# plt.show()


# # Calculate True Positive Rate (TPR), False Positive Rate (FPR), and Thresholds:
# fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label='>50K')

# roc_auc = auc(fpr, tpr)
# # Calculate AUC Score:
# display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
#                           estimator_name='(ROC) Curve')

# display.plot()
# plt.legend(loc="lower right")
# plt.plot([0, 1], [0, 1], 'k--')
# plt.title('Roc Curve of Gradient boosting')
# plt.show()



# precision, recall, _ = precision_recall_curve(y_true=y_test, y_score=y_pred,pos_label='>50K')
# disp = PrecisionRecallDisplay(precision=precision, recall=recall)
# disp.plot()
# plt.show()


# import pickle
# file = open('model','wb')
# pickle.dump(best_model,file)
# file.close()