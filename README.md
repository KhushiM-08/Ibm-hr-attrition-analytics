# 📊 IBM HR Attrition Analytics

Employee attrition prediction system built on the IBM HR Analytics dataset using Python, SQL, Excel, and Power BI.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat&logo=flask)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?style=flat&logo=scikit-learn)](https://scikit-learn.org)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-blue?style=flat&logo=mysql)](https://mysql.com)

---

## 📈 Results

| Model | ROC-AUC |
|-------|---------|
| Random Forest | 0.862 |
| Gradient Boosting | 0.854 |
| **Logistic Regression** | **0.870** |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python scripts/create_ibm_dataset.py
python scripts/train_model.py
python web/app.py
```


---

## 🗂️ What's Inside

| Folder | Contents |
|--------|---------|
| `scripts/` | Dataset generation + full ML pipeline + MySQL loader |
| `sql/` | Schema, 3 stored procedures, 12 analytical queries, Power BI view |
| `reports/` | 15 auto-generated matplotlib/seaborn charts |
| `data/` | IBM dataset, risk scores, 5-sheet Excel report |
| `web/` | Flask app — 5 pages, 9 REST API endpoints |

---

## 🛠️ Tech Stack

`Python` `scikit-learn` `SHAP` `Flask` `MySQL` `Excel` `Power BI` `Chart.js`

---

## 📌 Dataset
[IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — Kaggle · 1,470 employees · 35 features · 16.12% attrition rate

---

## 📄 License
MIT
