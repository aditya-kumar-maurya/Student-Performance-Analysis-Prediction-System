# Data Dictionary

| Column | Type | Purpose |
|---|---|---|
| student_id | Text | Unique student identifier |
| student_name | Text | Student name |
| age | Integer | Student age |
| gender | Category | Male/Female |
| department | Category | CSE/IT/ECE/ME/CE |
| semester | Integer | Semester 1-8 |
| attendance_percentage | Numeric | Attendance percentage |
| study_hours_per_day | Numeric | Average daily study hours |
| previous_exam_score | Numeric | Previous exam score |
| midterm_score | Numeric | Midterm score |
| assignment_percentage | Numeric | Assignment performance |
| practical_lab_score | Numeric | Practical/lab performance |
| sleep_hours_per_day | Numeric | Average daily sleep |
| online_learning_hours | Numeric | Online learning hours |
| previous_backlogs | Integer | Previous backlog count |
| internet_access | Category | Yes/No |
| quiz_score | Numeric | Quiz performance |
| class_participation_score | Numeric | Participation score; stored and analyzed |
| final_score | Numeric | Target variable |

The Random Forest model uses exactly 15 prediction features: age, gender, department, semester, attendance, study hours, previous exam, midterm, assignment, practical/lab, sleep, online learning, previous backlogs, internet access and quiz score.
