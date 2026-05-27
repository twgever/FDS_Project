# FDS Project

## Overview

This repository contains a collection of machine learning experiments developed for the *Foundations of Data Science* course during the Winter Semester 2023.

The project focuses on the analysis and prediction of student performance using multiple supervised learning approaches. The notebooks explore how different models behave on the same prediction task, with an emphasis on methodology, preprocessing, model evaluation, and comparison of results.

The repository is organized as a set of independent Jupyter notebooks, each dedicated to a specific machine learning technique.

---

## Project Structure

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

- `StudentPerformanceAnalysis.ipynb`  
  General exploratory analysis of the dataset and project introduction.

- `LinearRegression.ipynb`  
  Linear regression model implementation and evaluation.

- `KNNRegression.ipynb`  
  K-Nearest Neighbors regression experiments, including model selection for different values of `k`.

- `DecisionTrees.ipynb`  
  Decision tree based regression analysis, including handling of non-linear relationships.

- `MultiLogReg.ipynb`  
  Multinomial logistic regression experiments for classification-related tasks.

- `NeuralNetworks.ipynb`  
  Neural network experiments, including feature engineering and model optimization.

- `report/`  
  Contains figures and visual material used in the project report.

---

## Objectives

The project investigates different supervised learning techniques applied to educational performance data. The main objectives are:

- exploring the structure and characteristics of the dataset;
- preprocessing and transforming the data for machine learning tasks;
- comparing regression and classification approaches;
- evaluating model performance using standard metrics;
- studying the impact of model complexity and feature engineering.

---

## Technologies and Libraries

The implementation is primarily based on Python and Jupyter notebooks.

Main libraries used in the project include:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `PyTorch`
- `imbalanced-learn`

---

## Methods Used

The repository includes experiments with several machine learning techniques, including:

- Linear Regression
- K-Nearest Neighbors Regression
- Decision Trees
- Multinomial Logistic Regression
- Neural Networks

Additional topics addressed in the notebooks include:

- exploratory data analysis;
- feature engineering;
- oversampling techniques;
- hyperparameter selection;
- model evaluation and error analysis;
- visualization of results.

---

## Usage

### 1. Clone the repository

```bash
git clone <repository-url>
cd FDS_Project-main
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch imbalanced-learn notebook
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook of interest from the Jupyter interface.

---

## Notes

- The notebooks are intended primarily for academic and experimental purposes.
- Some notebooks may assume the presence of datasets or intermediate preprocessing steps defined elsewhere in the project.
- Figures used in the accompanying report are stored in the `report/` directory.

---

## Authors

Project developed for the *Foundations of Data Science* course, Winter Semester 2023.

Authors listed in the notebooks include:

- Tommaso Leonardi
- Arianna Paolini
- Stefano Saravalle
- Paolo Cursi
- Pietro Signorino
