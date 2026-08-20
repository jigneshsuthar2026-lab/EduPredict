import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#-------------------
#1. Create a dataset
#-------------------
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 1, 4, 6, 7, 3, 5, 8],
    "Attendance": [80, 85, 90, 95, 100, 75, 70, 65, 60, 85, 90, 95, 80, 75, 70],
    "Previous_marks": [70, 75, 80, 85, 90, 65, 60, 55, 50, 75, 80, 85, 70, 65, 60],
    "Assignments":[50, 55, 60, 65, 70, 75, 80, 85, 90, 45, 62, 78, 85, 58, 68],
    "Sleep_hours": [6, 7, 8, 5, 4, 9, 10, 3, 2, 7, 8, 6, 5, 4, 9],
    "Final_marks": [75, 80, 85, 90, 95, 70, 65, 60, 55, 80, 85, 90, 75, 70, 65]
}
df = pd.DataFrame(data)

print("Student Dataset:")
print(df)
#----------------------------
#2. Separate input and output
#----------------------------

X = df[["Study_Hours", "Attendance", "Previous_marks", "Assignments", "Sleep_hours"]]
y = df["Final_marks"]
#--------------------
#3. Split the dataset
#--------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#--------------------
#4. Test the ML model
#--------------------
model = LinearRegression()
model.fit(X_train, y_train)
#--------------------
#5. Test the model
#--------------------
predictions = model.predict(X_test)
print("\nActual Marks:")
print(y_test.values)
print("\nPredicted Marks:")
print(predictions.round(2))
#--------------------------
#6. Check model performance
#--------------------------
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("\nMean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))
#----------------------------------
#7. Predict marks for a new student
#----------------------------------
study_hours = float(input("\nEnter Study Hours: "))
attendance = float(input("Enter Attendance Percentage: "))
previous_marks = float(input("Enter Previous Marks: "))
assignments = float(input("Enter Assignments Marks: "))
sleep_hours = float(input("Enter Sleep Hours: "))

new_student = [[study_hours, attendance, previous_marks, assignments, sleep_hours]]
predicted_marks = model.predict(new_student)
print("\nPredicted Final Marks for the new student:", round(predicted_marks[0], 2))
#-----------------------------
#8. Give Performance Category
#-----------------------------
marks = predicted_marks[0]

if marks >= 90:
    category = "Excellent"
elif marks >= 75:
    category = "Very Good"
elif marks >= 60:
    category = "Good"
elif marks >= 40:
    category = "Average"
else:
    category = "Poor"
