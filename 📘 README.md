# 🧠 Advanced Salary Prediction ML Application

## 📋 Overview
The **Advanced Salary Prediction ML Application** is a GUI-based machine learning tool built using **Tkinter** and **scikit-learn**.  
It predicts an employee’s **salary** based only on **Years of Experience** using a **Linear Regression** model.  

This project demonstrates a complete **end-to-end ML workflow** — from data loading and training to prediction and visualization — inside a simple and user-friendly interface.

---

## 🚀 Features
✅ User-friendly **Tkinter GUI**  
✅ Accepts **CSV dataset** with only two columns: `YearsExperience` and `Salary`  
✅ **Automatic training and model evaluation** using scikit-learn  
✅ **Real-time salary prediction** for any input  
✅ **Graph visualization** (scatter + regression line)  
✅ **Save/Load model** using Pickle  
✅ Lightweight and easy for beginners

---

## 🧩 Project Structure
Advanced_Salary_Prediction/
│
├── app.py # Main application file (GUI + ML logic)
├── sample_dataset.csv # Example dataset (YearsExperience, Salary)
├── requirements.txt # Python dependencies
├── saved_model.pkl # Trained model (auto-generated after training)
└── README.md # Documentation
-------------------------------------------------------------


---

## ⚙️ Installation and Setup

### ️⃣  Prerequisites
- Python **3.8+**
- pip (Python package manager)
- Works on **Windows / macOS / Linux**

### ️⃣  Clone or Download the Project
```bash
git clone https://github.com/your-username/advanced-salary-prediction.git
cd advanced-salary-prediction

### ️⃣  Create a Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On macOS/Linux

### ️⃣  Install Dependencies

pip install -r requirements.txt



### 📦 Dependencies
| Library             | Purpose                                       |
| ------------------- | --------------------------------------------- |
| **pandas**          | Load and manage dataset                       |
| **numpy**           | Mathematical operations                       |
| **scikit-learn**    | Machine learning (Linear Regression, metrics) |
| **matplotlib**      | Visualization                                 |
| **tkinter**         | GUI framework                                 |
| **joblib / pickle** | Save and load model                           |

### ️⃣  requirements.txt

pandas
numpy
scikit-learn
matplotlib


🧠 Model Details
Input Dataset Format

Your CSV must include exactly two columns:

Column		Description					Example
YearsExperience	Number of years of professional experience	1.1, 2.3, 5.5
Salary		Corresponding salary				39343, 46205, 60000

✅ Example Dataset

YearsExperience,Salary
1.1,39343
1.3,46205
1.5,37731
2.0,43525
2.2,39891
3.0,56642
4.0,57189
5.0,66029
6.0,83088

🧮 Model Algorithm

Algorithm Used: Linear Regression

Feature: YearsExperience

Target: Salary

Evaluation Metrics

R² Score (model accuracy)

Mean Squared Error (MSE)

🧑‍💻 How to Use

Run the App

python app.py
Upload Dataset

Click “Upload Dataset”

Choose a CSV file with two columns: YearsExperience and Salary

Train the Model

Click “Train Model”

The app will show accuracy and training results

Predict Salary

Enter Years of Experience

Click “Predict” to view the estimated salary

Show Graph

Click “Show Graph” to visualize the regression trend

Save/Load Model

Use the buttons to store or reload the trained model (saved_model.pkl)

💾 Model Persistence
The trained model is saved as:

saved_model.pkl


You can reuse it later without retraining.
🎨 GUI Interface

The Tkinter interface includes:

Upload & Train buttons

Text inputs for prediction

Live metrics display

Matplotlib plot window for regression visualization

🧰 Future Enhancements

Add multiple regression algorithms (Decision Tree, Random Forest)

Add support for extra features (Age, Education, etc.)

Export predictions to Excel or PDF

Deploy as a web app using Streamlit or Flask

📚 Learning Outcome

This project teaches:

Linear Regression modeling

Dataset handling with Pandas

Data visualization with Matplotlib

GUI development with Tkinter

Model evaluation and persistence

👨‍💻 Author

Developed by: Akbar Pirzada
Focus: Data Science • Machine Learning • GUI-based ML Tools

Email: akbar.perzada@gmail.com
LinkedIn: https://linkedin.com/akbar.pirzada
