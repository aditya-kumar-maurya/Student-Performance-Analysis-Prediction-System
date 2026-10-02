# Student Performance Analysis & Prediction System

A complete fresher-friendly Python + Flask + SQLite + Machine Learning project for analyzing academic performance and predicting a student's final score.

## 1. Main Purpose

The main purpose of this project is to:
- analyze student academic performance;
- understand patterns using data analysis and visualization;
- predict a student's final score from academic information;
- help teachers identify students who may need additional support.

This project was made to understand how data cleaning, data analysis and machine learning can be applied to a real-life student dataset.

## 2. Technology Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SQLite
- SQL queries
- Scikit-learn
- Random Forest Regression
- Flask
- HTML/CSS/JavaScript

## 3. Role of Each Technology

| Technology | Use |
|---|---|
| Python | Overall project development |
| NumPy | Numerical calculations and numerical data |
| Pandas | Data cleaning, processing and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization and correlation heatmap |
| SQLite | Store student information and prediction history |
| SQL | SELECT, INSERT and aggregation queries |
| Scikit-learn | Train and evaluate the ML model |
| Random Forest Regression | Predict final score |
| Flask | Web application and model/frontend integration |

## 4. Dataset

The project contains **500 synthetic student records**.

### 15 prediction features
1. Age
2. Gender
3. Department
4. Semester
5. Attendance %
6. Study hours per day
7. Previous exam score
8. Midterm score
9. Assignment %
10. Practical/Lab score
11. Sleep hours per day
12. Online learning hours
13. Previous backlogs
14. Internet access
15. Quiz score

### Additional stored fields
- Student ID
- Student Name
- Class Participation Score
- Final Score (target variable)

`class_participation_score` is cleaned and included in analysis, while the model intentionally uses exactly the 15 prediction features listed above.

## 5. Data Cleaning

Pandas is used to:
- detect missing values;
- convert numeric columns to numeric types;
- fill numeric missing values using the median;
- fill categorical missing values using the mode;
- prepare the dataset for analysis and machine learning.

The raw CSV intentionally contains a few missing values so the cleaning step can be demonstrated.

## 6. Data Analysis and Visualization

The project generates:
- Department-wise average score — Bar Chart
- Study Hours vs Final Score — Scatter Plot
- Attendance vs Final Score — Scatter Plot
- Semester-wise average score — Line Chart
- Final Score Distribution — Histogram
- Performance Category — Pie Chart
- Academic Correlation Heatmap

## 7. Machine Learning

Target variable:
`final_score`

Model:
**Random Forest Regression from Scikit-learn**

The model is trained using a train/test split. The pipeline handles:
- numeric values;
- missing values;
- categorical encoding;
- Random Forest Regression.

Evaluation metrics printed by `scripts/train_model.py`:
- MAE
- RMSE
- R²

These metrics are for the included synthetic dataset and should not be presented as real-world model performance.

## 8. Flask Web Application

The dashboard follows this flow:

First Screen
↓
Hero + Project Image
↓
Dashboard Overview
↓
Performance Analytics
↓
AI Score Prediction
↓
Student Analysis
↓
Data Insights
↓
Student Records
↓
Prediction History
↓
Technology + Footer

### Main features
- Dashboard metrics
- Academic charts
- Final score prediction form
- Student search by ID
- Complete student profile
- Student records: view and search (SQLite CRUD-enabled)
- CSV import
- Prediction history
- SQLite database
- Random Forest model connected to Flask

## 9. How to Run

### Step 1: Open terminal
Go to the project folder.

### Step 2: Create a virtual environment
```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

### Step 3: Install packages
```bash
pip install -r requirements.txt
```

### Step 4: Run the project
The ZIP already contains the dataset, SQLite database, charts and trained model, so you can directly run:

```bash
python run.py
```

Open:
`http://127.0.0.1:5000`

### Rebuild everything from scratch
If you want to recreate the data/model:
```bash
python scripts/generate_data.py
python scripts/train_model.py
python scripts/init_db.py
python scripts/generate_charts.py
python run.py
```

## 10. Project Explanation for Interview

> "My project is Student Performance Analysis and Prediction System. The main purpose of this project is to analyze students' academic performance and predict their final score based on different academic factors.
>
> I used Python for the overall development. I used NumPy for numerical calculations and Pandas for data cleaning, processing and analysis.
>
> For visualization, I used Matplotlib and Seaborn to create charts such as department-wise average score, study hours versus final score, attendance versus final score and performance category.
>
> I used SQLite to store student information and SQL SELECT queries to read student records and dashboard metrics. I also use an INSERT query to store prediction history.
>
> For prediction, I used a Random Forest Regression model from Scikit-learn. The input is student academic information and the output is the predicted final score.
>
> Finally, I used Flask to create a web application where users can view the dashboard, search student records and predict a student's final score."

## 11. Simple Project Workflow

Student Data
→ Data Cleaning
→ Data Processing
→ Data Analysis
→ Visualization
→ SQLite + SQL
→ Machine Learning
→ Random Forest Regression
→ Final Score Prediction
→ Flask Web Application

## 12. Important Note

The dataset is synthetic and created for educational/demo purposes. It is not real student data.


## SQLite Student Management
The project does not use MySQL and does not have a login page. Student records are managed directly in SQLite with SQL CRUD operations:
- Create: Add Student (`INSERT`)
- Read: View/Search Student (`SELECT`)
- Update: Edit Student (`UPDATE`)
- Delete: Delete Student (`DELETE`)
- CSV Import: Pandas reads the CSV and inserts/replaces rows in SQLite.

The dashboard, prediction history, and student analysis all use the same `data/student_performance.db` database.
## UI Theme

The dashboard uses a dark-mode interface with colorful Matplotlib/Seaborn charts for better contrast and readability.
