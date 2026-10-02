# Employee Attrition Prediction 📊

### **Machine Learning-Based HR Analytics Application**

Employee Attrition Prediction is an interactive machine learning application that estimates the probability of employee attrition from employee and workplace-related characteristics. The project includes data preprocessing, model training and comparison, interactive prediction, model-sensitivity indicators, and a Streamlit-based analytics dashboard.

---

## 🚀 Live Demo

[**Open Employee Attrition Predictor**](https://employee-attrition-gebnrgd3snvkam5pifkg5w.streamlit.app/)

---

## ✨ Key Features

* **Employee attrition probability prediction**
* **Logistic Regression, Decision Tree, and Random Forest comparison**
* **One-Hot Encoding for categorical variables**
* **80/20 stratified train-test split**
* **Estimated attrition probability with risk interpretation**
* **Model-sensitivity indicators for individual predictions**
* **Prediction history within the current session**
* **Interactive analytics dashboard**
* **CSV export of prediction history**

---

## 🗂️ Dataset

The project uses the **IBM HR Analytics Employee Attrition & Performance** dataset.

| Property | Details |
| :--- | :--- |
| **Records** | 1,470 |
| **Original Columns** | 35 |
| **Target Variable** | `Attrition` |
| **Processed Features** | 44 |

The target variable contains two classes:
```text
Yes → Employee attrition
No  → No employee attrition
```

---

## 🧹 Data Preprocessing

The following non-predictive or constant columns were removed before training:
* `EmployeeCount`
* `EmployeeNumber`
* `Over18`
* `StandardHours`

Categorical variables were converted into numerical features using One-Hot Encoding with `drop_first=True`. The categorical variables include:
* BusinessTravel
* Department
* EducationField
* Gender
* JobRole
* MaritalStatus
* OverTime

After preprocessing, the final feature set contains **44 features**.

---

## 📊 Model Performance

Three models were trained and evaluated on the same preprocessed dataset.

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 74.49% | 34.09% | 63.83% | 44.44% |
| **Decision Tree** | 76.53% | 34.72% | 53.19% | 42.02% |
| **Random Forest** | **80.61%** | **43.06%** | **65.96%** | **52.10%** |

### **Deployed Model: Random Forest**
The Random Forest model is used for live predictions in the deployed application. Logistic Regression and Decision Tree were trained and evaluated for comparison.

---

## 🔮 Live Prediction

The application takes employee-related inputs and generates an Estimated Attrition Probability using the trained Random Forest model.

```text
User Input 
   ↓ 
Feature Encoding 
   ↓ 
Feature Alignment 
   ↓ 
Random Forest 
   ↓ 
Estimated Attrition Probability 
   ↓ 
Risk Interpretation
```

### **Risk Interpretation**
* **Low Risk:** < 40%
* **Moderate Risk:** 40% – < 50%
* **High Risk:** ≥ 50%

*Note: These thresholds are application-defined interpretation bands and are not organizationally calibrated risk categories.*

---

## 🔍 Model-Sensitivity Indicators

The application provides model-sensitivity indicators to help inspect how individual input changes affect the model output. 

For a prediction, one input characteristic is changed at a time while the other inputs remain unchanged. The resulting change in Random Forest probability is then calculated. These indicators describe the trained model's response to the supplied profile. They are not causal explanations of individual employee behavior.

---

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Programming** | Python |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Plotly |
| **Application** | Streamlit |
| **Model Serialization** | Joblib |
| **Version Control** | Git, GitHub |
| **Deployment** | Streamlit Community Cloud |

---

## 📁 Project Structure

```text
Employee_Attrition_Project/
│
├── app.py
├── train_models.py
├── rf.pkl
├── dt.pkl
├── lr.pkl
├── columns.pkl
├── job_role_proxy.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

### **Important Files**
* **`app.py`** — Streamlit application, prediction workflow, and dashboard
* **`train_models.py`** — Data preprocessing, model training, evaluation, and model saving
* **`rf.pkl`** — Trained Random Forest model used for live predictions
* **`dt.pkl`** — Trained Decision Tree model
* **`lr.pkl`** — Trained Logistic Regression model
* **`columns.pkl`** — Final 44-feature structure used for inference
* **`job_role_proxy.pkl`** — Application-level mapping for roles not present in the original dataset
* **`requirements.txt`** — Python dependencies

---

## ⚙️ Run Locally

### **1. Clone the repository**
```bash
git clone https://github.com/anushkasingh8002/employee-attrition.git
cd employee-attrition
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the application**
```bash
streamlit run app.py
```
---
## **🧠 Retraining the Models**
To retrain the models using the original dataset:
```bash
python train_models.py
```
The training script performs preprocessing, One-Hot Encoding, stratified train-test splitting, model training, evaluation, comparison, and model serialization.

---

## ⚠️ Limitations

* The model is trained on the IBM HR Analytics dataset and may not generalize to every organization.
* The dataset represents a specific HR dataset and should not be treated as representative of all employees or workplaces.
* Risk thresholds are application-defined and are not organizationally calibrated.
* Model-sensitivity indicators describe model behavior and should not be interpreted as causal explanations.
* The original dataset does not contain the roles **AI/ML Engineer** or **Software Developer**. These application options use an application-level proxy mapping to an available training category rather than adding new training records.

---

## 🔒 Disclaimer

This project is intended for educational and demonstration purposes. The predictions are statistical model outputs and should not be used as the sole basis for employment, hiring, promotion, retention, or other HR decisions.

---

## 👩‍💻 Developer

**Anushka Singh**  
*B.Tech Computer Science & Engineering*  
*Machine Learning & Software Development*  
[GitHub](https://github.com/anushkasingh8002)

---

## 📌 Project Status

* **Deployment:** Deployed and functional
* **Dataset:** IBM HR Analytics Employee Attrition & Performance
* **Records:** 1,470
* **Processed Features:** 44
* **Models:** Logistic Regression, Decision Tree, Random Forest
* **Deployed Model:** Random Forest
* **Application Framework:** Streamlit
* **Live Demo:** [Open Live Demo](https://employee-attrition-gebnrgd3snvkam5pifkg5w.streamlit.app/)
