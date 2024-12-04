# Getting Data:
# The data used in this project is from a dataset in this url : <a>https://www.kaggle.com/datasets/wenruliu/adult-income-dataset</a>

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
    ['Order Date','Ship Date','Ship Mode','Segment','City', 'State','Country','Market','Region','Category','Sub-Category','Sales','Quantity','Discount','Profit','Shipping Cost','Order Priority'		
]]

print(data.head(10))

# Remove unwanted columns
# we will not use all rows  because we are interseted only in african market 


data = data[data.isin(['Africa']).any(axis=1)]
print(data.head(10))
print("number of rows is : ",data.count())


# # EDA
# Understanding data
# All columns exist in our data

data.columns
"""
# Age and income (Histogram)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

# Bigger than 50K
big_sal = data[data['income']=='>50K']
big_sal['age'].plot(kind='hist',ax=axes[0][0],label='Age histogram',bins=20,
                      title='Age of employees which their salary bigger than 50K',
                      color='red', edgecolor='black')

axes[0][0].set_xlabel('age')
axes[0][0].set_ylabel('count')
axes[0][0].legend()
for container in axes[0][0].containers:
    axes[0][0].bar_label(container, label_type='edge')
axes[0][0].plot()

big_sal['age'].plot(kind='kde',ax=axes[1][0],label='Age KDE',
                      color='red')
axes[1][0].set_xlabel('Age')
axes[1][0].set_ylabel('density')
axes[1][0].plot()

# lesser than 50K
small_sal = not_null_data[not_null_data['income']=='<=50K']
small_sal['age'].plot(kind='hist', ax=axes[0][1],label='Age histogram',bins=10,
                      title='Age of employees which their salary smaller than 50K',
                      color='blue')
axes[0][1].set_xlabel('Age')
axes[0][1].set_ylabel('Count')
axes[0][1].legend()
for container in axes[0][1].containers:
    axes[0][1].bar_label(container, label_type='edge')
axes[0][1].plot()

small_sal['age'].plot(kind='kde',ax=axes[1][1],label='Age KDE',
                      color='blue')
axes[1][1].set_xlabel('Age')
axes[1][1].set_ylabel('density')
axes[1][1].plot()


plt.tight_layout()
plt.show()

# ## Education per Income (Bar)
#   


# Education levels its ordered based on level of education
educations = ['Preschool', '1st-4th','5th-6th',
              '7th-8th', '9th', '10th',
              '11th', '12th', 'HS-grad',
              'Assoc-voc', 'Some-college', 'Assoc-acdm',
               'Prof-school', 'Bachelors', 'Masters','Doctorate',
                ]
# Group by occupation and salary and count occurrences
Education_counts = data.groupby(['education', 'income']).size().unstack(fill_value=0)

# Plot the bar chart
Education_counts['total'] = Education_counts.sum(axis=1)
Education_counts_sorted = Education_counts.loc[educations].drop(columns='total')


ax = Education_counts_sorted.plot(kind='bar', stacked=False, figsize=(10, 6), color=['blue', 'orange'])
# Add labels and title
plt.title('number of peoples Salaries based on jobs title')
plt.xlabel('Occupation')
plt.ylabel('Number')
plt.xticks(ha='right')
plt.legend(title='Salary')

for container in ax.containers:
    ax.bar_label(container, label_type='edge')
# Show the plot
plt.tight_layout()
plt.show()

"""
# ## Work class and income (Pie)
# Count the occurrences of each order priority
order_count = data['Order Priority'].value_counts()

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(order_count, labels=order_count.index,autopct='%1.1f%%', colors=['blue', 'yellow', 'red', 'green'])
plt.title('Order Priority Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show the plot
plt.show()

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
plt.show()
"""
# import library
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, auc, precision_recall_curve, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

from sklearn.feature_selection import  RFECV


# Apply random under sampler


from collections import Counter
X = data.drop(['income'],axis=1)
y = data["income"]
print(Counter(y))

rus = RandomUnderSampler(random_state=40)
X, y = rus.fit_resample(X,y)
print(Counter(y))


print(X.shape)
print(y.shape)

# Splitting data into train and test 


# Split data into Train and validation set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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
       ("cat", OneHotEncoder(), categorical_features),
       ("num", StandardScaler(), numerical_features),
   ]
)

#  Use RFECV as Feauture selector

# *Pipeline* is a series of data processing steps. Instead of creating new instances of each element in our model, we can simply put all those tasks into one pipeline.


pipeline = make_pipeline(preprocessor,
                         RFECV(estimator=DecisionTreeClassifier(random_state=42), verbose=1, step=1, cv=5, scoring='accuracy'),# verbose : Show iteration 
                         GradientBoostingClassifier(random_state=42),
                         verbose=True,
                         
                   )
pipeline



# Fit the model on the training data
print('Fit the model')
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Predict Probability on the test set
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

# Generate classification report
report = classification_report(y_test, y_pred)



# Define the parameter grid for Gradient Boosting Classifier
param_grid = {
    'gradientboostingclassifier__n_estimators': [50, 100, 200],
    'gradientboostingclassifier__learning_rate': [0.1, 0.05, 0.01],
    'gradientboostingclassifier__max_depth': [3, 5, 7]
}



# Initialize GridSearchCV with the pipeline and parameter grid
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search
# Fit the grid search on your data
# X_train and y_train should be your training data
grid_search.fit(X_train, y_train)



# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

best_params
# best_model

results_df = pd.DataFrame(grid_search.cv_results_)
results_df[['mean_test_score','mean_fit_time','rank_test_score','params']].sort_values(by='rank_test_score',ascending=True)

# ## Model results


# Predict on the test set
y_pred = best_model.predict(X_test)


# Predict Probability on the test set
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

# Generate classification report
report = classification_report(y_test, y_pred)



print("\nClassification Report:")
print(report)



# Predict on the test set
y_pred = pipeline.predict(X_test)

# Predict Probability on the test set
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

# Generate classification report
report = classification_report(y_test, y_pred)



print("\nClassification Report:")
print(report)



cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
cm_show = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['>50K', '<=50K'])
cm_show.plot(cmap='Blues')
plt.title('Confusion matrix')
plt.show()


# Calculate True Positive Rate (TPR), False Positive Rate (FPR), and Thresholds:
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label='>50K')

roc_auc = auc(fpr, tpr)
# Calculate AUC Score:
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                          estimator_name='(ROC) Curve')

display.plot()
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Roc Curve of Gradient boosting')
plt.show()



precision, recall, _ = precision_recall_curve(y_true=y_test, y_score=y_pred_prob,pos_label='>50K')
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.show()


import pickle
file = open('model','wb')
pickle.dump(best_model,file)
file.close()
"""