# End-to-End AI Carbon Emission Prediction using Machine Learning

## Introduction

Artificial Intelligence (AI) models require significant computational resources during training, resulting in substantial energy consumption and carbon emissions. As AI adoption continues to grow across industries, understanding and predicting its environmental impact has become increasingly important. This project leverages Machine Learning techniques to estimate the carbon emissions generated during AI model training based on model characteristics and training parameters.

## Motivation

The primary goal of this project is to develop a data-driven solution that predicts the carbon footprint of AI models. By analyzing factors such as model size, training time, hardware configuration, cloud provider, and training hyperparameters, this system helps researchers and organizations estimate environmental impact before training large-scale AI models. This promotes sustainable AI development and encourages environmentally responsible decision-making.

---

# Project Overview

The project follows a complete Machine Learning pipeline:

- Data Collection
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Scaling & Encoding
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Prediction System

---

# Features

- Predicts AI model carbon emissions
- Handles categorical and numerical features
- Data preprocessing and feature engineering
- Trained using multiple regression algorithms
- Hyperparameter tuning using GridSearchCV
- Performance evaluation using MAE, MSE, RMSE, and R² Score
- Easy-to-use prediction interface

---

# Dataset

The project uses a synthetic dataset containing AI model training metrics from **2018–2025**.

### Features include:

- Model Type
- Model Size (Million Parameters)
- Training Framework
- Number of Epochs
- Batch Size
- Learning Rate
- GPU Type
- Cloud Provider
- Training Time (Hours)
- Accuracy
- Precision
- Recall
- F1 Score
- Carbon Emission (Target Variable)

---

# Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

# Machine Learning Models

The following regression algorithms were implemented and compared:

- Linear Regression
- Ridge Regression
- K-Nearest Neighbors Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regressor (SVR)
- AdaBoost Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

The best-performing model was selected based on evaluation metrics.

---

# Model Evaluation Metrics

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# Installation Guide

## Prerequisites

Before running the project, install the following libraries:

- Python 3.10+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Carbon-Emission-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd AI-Carbon-Emission-Prediction
```

### 3. Create a Virtual Environment (Optional)

```bash
conda create -n carbon_env python=3.10 -y
```

### 4. Activate the Environment

```bash
conda activate carbon_env
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook and run all cells.

---

# Project Structure

```
AI-Carbon-Emission-Prediction/
│
├── data/
│   └── synthetic_ai_model_metrics_2018_2025.xlsx
│
├── notebooks/
│   └── project-code.ipynb
│
├── models/
│   ├── trained_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Results

The trained Machine Learning model can accurately predict the carbon emissions of AI models using training and hardware-related features. The project demonstrates how predictive analytics can support environmentally sustainable AI development.

---

# Future Improvements

- Deploy the model using Streamlit or Flask
- Integrate real-world AI training datasets
- Add carbon emission visualization dashboard
- Support deep learning-based prediction models
- Cloud deployment using AWS or Azure

---

# Contributing

Contributions are welcome!

If you have suggestions for improving the project, feel free to:

- Fork the repository
- Create a feature branch
- Commit your changes
- Submit a Pull Request

---

# Acknowledgements

This project was developed as part of a Machine Learning learning journey focused on sustainable AI. The synthetic dataset was used for educational and research purposes. Special thanks to the open-source Python community and the developers of Scikit-learn, XGBoost, Pandas, NumPy, and Matplotlib.

---

# Author

**Nandana T K**

M.Sc. Physics | Data Science & Machine Learning Enthusiast

GitHub: https://github.com/nandana2803-star

LinkedIn: www.linkedin.com/in/nandana-tk-
