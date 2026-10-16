# ML-Project-and-Learning

Welcome to my journey learning machine learning! This repository serves as a practical showcase of how ML algorithms work, the mathematics involved, structures, rules, and the overall fundamentals I am mastering through hands-on experience. 

*Note: This repository is an educational portfolio and learning journal rather than a standalone software application.*

---

## 🗺️ Repository Roadmap & Navigation

### 📈 [Linear Regression](./ML%20LINEAR%20REG)
* **Status:** Completed
* **Details:** Contains exercise files and the core application.
  * 📁 Practice and Exercises: [`./ML LINEAR REG/Practice and ...`](./ML%20LINEAR%20REG)
  * 📁 Final Implementation: [`./ML LINEAR REG/Project`](./ML%20LINEAR%20REG/Project)

### 📊 [Logistic Regression](./ML%20LOG%20REG)
* **Status:** In Progress / Researching accuracy optimization.
* **Details:** Implementation files, datasets, and complete project documentation are housed in this dedicated folder.
* **Core Script:** [`LogReg.py`](./ML%20LOG%20REG/LogReg.py)
* **Dataset:** [`student_performance_logistic.csv`](./ML%20LOG%20REG/student_performance_logistic.csv)
* **Documentation:** 
  * 📝 Plain Text: [`MLDocumentation.txt`](./ML%20LOG%20REG/MLDocumentation.txt) | [`logreg structure.txt`](./ML%20LOG%20REG/logreg%20structure.txt)
  * 📄 Word Doc: [`Log Reg Documentation.docx`](./ML%20LOG%20REG/Log%20Reg%20Documentation.docx)

### 🛠️ Machine Learning in C++
* **Status:** Upcoming / Just started
* **Details:** More information and source files coming soon!

---

## 📋 Project Breakdown: Student Performance Classifier
The main project in the **`ML LOG REG`** folder uses a **Logistic Regression** model optimized with `PolynomialFeatures` and `StandardScaler` to predict student exam outcomes. 

### Key Project Components:
* **Features Matrix:** Built using student data tracking study hours, sleep habits, attendance percentages, and previous grades.
* **Data Processing:** Implements a stratified train/test split to maintain balanced target distributions.
* **Evaluation Metrics:** Outputs overall model accuracy alongside automated target bin-counting for prediction validation.
