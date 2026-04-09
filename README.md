# 🎓 Student Burnout Predictor

A machine learning web app that predicts student burnout level — **Low**, **Medium**, or **High** — based on 19 features covering sleep habits, stress levels, academic pressure, CGPA, and more.

---

## 📊 Dataset

**150,000 student records** from the [Student Mental Health & Burnout dataset on Kaggle](https://www.kaggle.com/).

> ⚠️ The CSV file is not included in this repo due to size. Download it from Kaggle and place it in the root directory as `student_mental_health_burnout.csv`.

**Features include:**
- Sleep hours & sleep quality
- Daily study hours & attendance
- Stress, anxiety, and depression scores
- Academic pressure & financial stress
- Social support & physical activity
- CGPA, course, year, internet quality

---

## 🔑 Key Insight

Students sleeping **fewer than 6 hours** are **40% more likely** to fall into the High Burnout category compared to those sleeping 7–8 hours. Sleep isn't a luxury — it's fuel.

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Models | Scikit-learn, XGBoost, LightGBM |
| Ensemble | Voting Classifier (top 3 models) |
| Performance | F1-score > 90% on test set |
| Web App | Streamlit |
| Deployment | ngrok (from Google Colab) |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/student-burnout-predictor.git
cd student-burnout-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Place student_mental_health_burnout.csv in the root directory

# 4. Train the model (run the Colab notebook first)
# This generates model_artifacts/ with best_model.pkl, scaler.pkl, feature_meta.json

# 5. Run the app
streamlit run app.py
```

---

## 🗂️ Project Structure

```
student-burnout-predictor/
│
├── app.py                                # Streamlit web app
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
├── .gitignore                            # Excludes CSV + cache files
└── Student_Burnout_predictor_App.ipynb   # Notebook for EDA, data preprocessing, model building, and experimentation
```

---

## 🧠 Models Trained

| Model | Notes |
|---|---|
| Random Forest | 200 estimators |
| Extra Trees | 200 estimators |
| XGBoost | 300 estimators, depth 8 |
| LightGBM | 300 estimators, 95 leaves |
| Logistic Regression | C=5, lbfgs solver |
| **Voting Ensemble** | **Best 3 models combined** |

---

## 📱 App Pages

- **🔮 Predict** — fill in your details and get an instant burnout prediction with probability breakdown
- **🏆 Model Results** — compare all trained models with confusion matrices and feature importance charts

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT](LICENSE)
