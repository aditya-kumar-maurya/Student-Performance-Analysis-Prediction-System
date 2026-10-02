# Fresher Interview Questions & Answers

### 1. What is your project?
"My project is Student Performance Analysis and Prediction System. It analyzes student academic performance and predicts the final score using machine learning."

### 2. What is the main purpose?
"The purpose is to understand student performance from academic data and predict the expected final score."

### 3. What is the input and output?
"Input is student academic information such as attendance, study hours, previous score, midterm, assignment, practical and quiz scores. Output is the predicted final score."

### 4. Why did you use Pandas?
"I used Pandas for data cleaning, processing and analysis."

### 5. Why did you use NumPy?
"I used NumPy for numerical calculations and working with numerical data such as scores, percentages and averages."

### 6. Why did you use Matplotlib and Seaborn?
"I used them to visualize student data and understand patterns through charts."

### 7. Which charts did you create?
"Department-wise average score bar chart, study hours versus final score scatter plot, attendance versus final score scatter plot, semester trend, score distribution and performance category pie chart. I also created a correlation heatmap."

### 8. Which database did you use?
"I used SQLite because it is lightweight, serverless and easy to use in a Python student-level project."

### 9. How did you manage the database?
"I used SQL queries such as SELECT, INSERT, UPDATE, DELETE and GROUP BY queries."

### 10. Which machine learning algorithm did you use?
"I used Random Forest Regression from Scikit-learn."

### 11. Why regression?
"Because the target is a numerical value, the student's final score."

### 12. Why Random Forest Regression?
"It can learn non-linear relationships and can work with multiple student-related features. It is also simple to use with Scikit-learn."

### 13. What is Scikit-learn?
"Scikit-learn is a Python machine learning library used for preprocessing, model training and model evaluation."

### 14. What is Flask?
"Flask is a lightweight Python web framework. I used it to create the web application and connect the trained machine learning model with the frontend."

### 15. Explain the project workflow.
"First I collected the synthetic student dataset. Then I cleaned and processed the data using Pandas and NumPy. After that I analyzed and visualized the data using Matplotlib and Seaborn. Then I prepared the data for machine learning, trained a Random Forest Regression model and evaluated it. Finally I connected the model with Flask so a user can enter student information and get a predicted final score."

### 16. How does the prediction work?
"The user enters the student's academic information. Flask receives the form data, converts it into a Pandas DataFrame and sends it to the trained Random Forest pipeline. The model returns the predicted final score, which is then shown on the web page and saved in prediction history."

### 17. How did you handle missing values?
"I used Pandas. For numeric columns I used the median, and for categorical columns I used the mode."

### 18. What problem does the project solve?
"It provides a simple way to analyze student performance and estimate the final score. It can also help identify students who may need additional academic support."

### 19. Is your data real?
"No. The included 500 records are synthetic data created for educational and demonstration purposes."

### 20. What would you improve in the future?
"I could add authentication, role-based access for teachers/admins, more real-world data, model comparison, better feature engineering, deployment and automated reports."
