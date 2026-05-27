# FDS Project

> Comparative machine learning experiments on student performance datasets.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Networks-red)

---

## 📖 Overview

This repository contains coursework developed for the **Foundations of Data Science** course during the Winter Semester 2023.

The project explores different supervised learning techniques applied to educational performance data. The focus is primarily methodological: comparing models, preprocessing strategies, and evaluation procedures rather than building a production-oriented system.

The repository is organized as a collection of Jupyter notebooks, each dedicated to a specific machine learning approach.

---

## 📁 Repository Structure

```text
FDS_Project-main/
│
├── StudentPerformanceAnalysis.ipynb
├── LinearRegression.ipynb
├── KNNRegression.ipynb
├── DecisionTrees.ipynb
├── MultiLogReg.ipynb
├── NeuralNetworks.ipynb
│
└── report/
    ├── figures/
    └── *.png
```

### Main Notebooks

| Notebook | Description |
|---|---|
| `StudentPerformanceAnalysis.ipynb` | Exploratory analysis and preprocessing |
| `LinearRegression.ipynb` | Linear regression experiments |
| `KNNRegression.ipynb` | K-Nearest Neighbors regression analysis |
| `DecisionTrees.ipynb` | Decision tree models and comparisons |
| `MultiLogReg.ipynb` | Multinomial logistic regression experiments |
| `NeuralNetworks.ipynb` | Neural network implementation and optimization |

---

## 🎯 Objectives

The project investigates:

- exploratory data analysis (EDA);
- preprocessing and feature engineering techniques;
- regression and classification workflows;
- comparison between classical ML methods and neural approaches;
- evaluation of model performance using standard metrics.

---

## 🛠 Technologies

The implementation is based on **Python** and **Jupyter notebooks**.

### Main Libraries

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `PyTorch`
- `imbalanced-learn`

---

## 📊 Methods

The repository includes experiments with several machine learning techniques:

- Linear Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Multinomial Logistic Regression
- Feedforward Neural Networks

Additional topics addressed in the notebooks include:

- feature scaling;
- oversampling techniques;
- hyperparameter tuning;
- model validation;
- visualization of results.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd FDS_Project-main
```

(Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch imbalanced-learn notebook
```

---

## 🚀 Usage

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open the notebook of interest from the Jupyter interface.

Suggested reading order:

1. `StudentPerformanceAnalysis.ipynb`
2. `LinearRegression.ipynb`
3. `KNNRegression.ipynb`
4. `DecisionTrees.ipynb`
5. `MultiLogReg.ipynb`
6. `NeuralNetworks.ipynb`

---

## 📝 Notes

- The repository was developed primarily for academic purposes.
- Some notebooks may depend on preprocessing steps performed in previous notebooks.
- Figures and report material are stored in the `report/` directory.

---

## 👥 Authors

Developed for the **Foundations of Data Science** course (Winter Semester 2023).

- Tommaso Leonardi
- Arianna Paolini
- Stefano Saravalle
- Paolo Cursi
- Pietro Signorino
