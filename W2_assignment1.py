import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('/Users/ankitabehera/Downloads/Dataset 2.csv')

# Part A: Dataset Understanding

#1) Load the dataset and display the first five records
print(df.head())

#2) Determine the number of rows and columns in the dataset.
print("Rows and Columns:", df.shape)

#3) Display all column names.
print(df.columns)

#4) Identify numerical and categorical features.
print("Numerical Columns:")
print(df.select_dtypes(include=np.number).columns)

print("\nCategorical Columns:")
print(df.select_dtypes(include='object').columns)

#5) Check whether the dataset contains missing values.
print(df.isnull().sum())



# Part B: Exploratory Data Analysis

#6) Calculate the average age of users.
print("Average Age:", df["Age"].mean())

#7) Determine the average watch hours per week.
print("Average Watch Hours:", df["WatchHoursPerWeek"].mean())

#8) Find the average monthly spending of users.
print("Average Monthly Spend:", df["MonthlySpend"].mean())

#9) Count the number of users in each subscription category.
print(df["SubscriptionType"].value_counts())

#10) Determine the percentage of users who renewed their subscriptions.
renew_percentage = (
    (df["SubscriptionRenewed"] == "Yes").mean()
) * 100

print("Renewal Percentage:", renew_percentage)



# Part C: Data Preparation

#11) Convert categorical features into numerical form.
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["SubscriptionType"] = le.fit_transform(df["SubscriptionType"])
df["FavoriteGenre"] = le.fit_transform(df["FavoriteGenre"])
df["SubscriptionRenewed"] = le.fit_transform(df["SubscriptionRenewed"])

print(df.head())

#12) Define the feature set (X) and target variable (y) for subscription renewal prediction.
X = df.drop("SubscriptionRenewed", axis=1)
y = df["SubscriptionRenewed"]

print(X.head())
print(y.head())

#13) Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)



# Part D: Decision Tree Classification

#14) Train a Decision Tree model to predict whether a user will renew their subscription.
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

#15) Evaluate the model using accuracy.
y_pred_dt = dt_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", accuracy_dt)

#16) Generate and interpret the confusion matrix.
cm = confusion_matrix(y_test, y_pred_dt)

print("Confusion Matrix:")
print(cm)

print("TN =", cm[0][0])
print("FP =", cm[0][1])
print("FN =", cm[1][0])
print("TP =", cm[1][1])



# Part E: K-Nearest Neighbors (KNN)

#17) Train a KNN classifier with K = 5.
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

#18) Compare the accuracy of KNN with the Decision Tree model.
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print("Decision Tree Accuracy:", accuracy_dt)
print("KNN Accuracy:", accuracy_knn)

if accuracy_knn > accuracy_dt:
    print("KNN performs better.")
else:
    print("Decision Tree performs better.")



# Part F: Linear Regression

#19) Train a Linear Regression model to predict monthly spending.
X_reg = df.drop("MonthlySpend", axis=1)
y_reg = df["MonthlySpend"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

lr_model = LinearRegression()

lr_model.fit(X_train_reg, y_train_reg)

#20) Predict the monthly spending for a new user and interpret the result.
new_user = pd.DataFrame({
    'UserID':[800],
    'Age':[25],
    'Gender':[1],
    'SubscriptionType':[1],
    'WatchHoursPerWeek':[15],
    'DevicesUsed':[2],
    'FavoriteGenre':[2],
    'AdClicks':[10],
    'SubscriptionRenewed':[1]
})

prediction = lr_model.predict(new_user)

print("Predicted Monthly Spending:", prediction[0])


