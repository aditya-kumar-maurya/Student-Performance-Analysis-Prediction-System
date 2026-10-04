# Student Performance Analysis & Prediction System

A **Python + Flask + PostgreSQL + Machine Learning** web application for analyzing student academic performance and predicting a student's final score.

The project combines **Data Analysis, Data Visualization, Machine Learning, SQL, PostgreSQL, Supabase and Flask** into one practical academic project.

---

## 1. Project Overview

The system is designed to analyze student academic data and predict the final score using Machine Learning.

### Main Features

* Student performance analysis
* Data visualization
* Final score prediction
* Student CRUD operations
* CSV data import
* Performance dashboard
* Prediction history
* PostgreSQL database

---

## 2. Main Objectives

* Clean and preprocess student data
* Perform Exploratory Data Analysis
* Create meaningful visualizations
* Train a Machine Learning model
* Predict student final scores
* Store student and prediction data in PostgreSQL
* Provide a web-based interface using Flask

---

## 3. Technology Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Random Forest Regression
* Flask
* PostgreSQL
* Supabase
* SQL
* HTML
* CSS
* JavaScript
* Git & GitHub

---

## 4. Role of Technologies

| Technology    | Purpose                         |
| ------------- | ------------------------------- |
| Python        | Backend and project development |
| NumPy         | Numerical processing            |
| Pandas        | Data cleaning and analysis      |
| Matplotlib    | Data visualization              |
| Seaborn       | Statistical visualization       |
| Scikit-learn  | Machine Learning                |
| Random Forest | Final score prediction          |
| Flask         | Web application backend         |
| PostgreSQL    | Data storage                    |
| Supabase      | Cloud PostgreSQL                |
| SQL           | Database operations             |
| HTML/CSS/JS   | Frontend                        |
| Git/GitHub    | Version control                 |

---

## 5. Dataset

The project contains **500 synthetic student records**.

### Machine Learning Features

The model uses 15 input features:

1. Age
2. Gender
3. Department
4. Semester
5. Attendance Percentage
6. Study Hours Per Day
7. Previous Exam Score
8. Midterm Score
9. Assignment Percentage
10. Practical/Lab Score
11. Sleep Hours Per Day
12. Online Learning Hours
13. Previous Backlogs
14. Internet Access
15. Quiz Score

### Additional Fields

* Student ID
* Student Name
* Class Participation Score
* Final Score

`Final Score` is the target variable.

---

## 6. Data Cleaning & Preprocessing

Pandas is used for:

* Handling missing values
* Converting data types
* Handling numerical and categorical data
* Preparing features for Machine Learning

---

## 7. Exploratory Data Analysis

EDA is performed to understand relationships between student performance factors.

Examples:

* Attendance vs Final Score
* Study Hours vs Final Score
* Department-wise Performance
* Semester-wise Performance
* Score Distribution
* Feature Correlation

---

## 8. Data Visualization

The project includes:

* Bar Chart
* Scatter Plot
* Line Chart
* Histogram
* Pie Chart
* Correlation Heatmap

These charts help understand student performance patterns.

---

# 9. Machine Learning

### Target Variable

```text
final_score
```

### Algorithm

The project uses:

```text
Random Forest Regression
```

The Machine Learning workflow is:

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Preprocessing
   ↓
Train-Test Split
   ↓
Random Forest Regression
   ↓
Model Evaluation
   ↓
Final Score Prediction
```

### Evaluation Metrics

The model is evaluated using:

* MAE
* MSE
* RMSE
* R² Score

---

# 10. Flask Web Application

Flask is used to integrate the Machine Learning model with the web application.

The application provides:

* Dashboard
* Performance Analytics
* Student Analysis
* Student Management
* CSV Import
* Score Prediction
* Prediction History

---

# 11. Database

The project uses **PostgreSQL** for storing:

* Student records
* Academic information
* Final scores
* Prediction history

**Supabase** is used for cloud-hosted PostgreSQL.

---

# 12. Project Workflow

```text
Student Data
     ↓
Data Cleaning
     ↓
EDA & Visualization
     ↓
Machine Learning
     ↓
Random Forest Model
     ↓
Flask Application
     ↓
PostgreSQL / Supabase
     ↓
Prediction & Dashboard
```

---

# 13. Installation

Clone the repository:

```bash
git clone https://github.com/your-username/student-performance-analysis.git
cd student-performance-analysis
```

Create virtual environment:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your PostgreSQL/Supabase database and environment variables.

Run the application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# 14. Project Structure

```text
student-performance-analysis/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
├── model/
├── scripts/
├── static/
└── templates/
```

---

# 15. Future Enhancements

* Model comparison
* Hyperparameter tuning
* Student risk prediction
* Automated recommendations
* PDF/Excel reports
* Authentication
* Cloud deployment
* Advanced dashboards

---

# 16. Learning Outcomes

Through this project, I gained practical experience in:

**Python, Pandas, NumPy, Data Analysis, Data Visualization, Machine Learning, Scikit-learn, SQL, PostgreSQL, Flask and Git/GitHub.**

---

## Author

**Aditya Kumar Maurya**

Computer Science Engineering Student

**Project:** Student Performance Analysis & Prediction System
