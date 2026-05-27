# FDS Project

> Machine learning experiments and comparative analysis on student performance datasets.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Networks-red)

---

## Overview

This repository contains coursework developed for the **Foundations of Data Science** course during the Winter Semester 2023.

The project investigates different supervised learning approaches applied to educational performance data. The main focus is not the development of a production-ready system, but rather the comparative study of machine learning techniques, preprocessing strategies, and evaluation methodologies.

The repository is structured as a collection of Jupyter notebooks, each dedicated to a specific model or methodological approach.

---

## Repository Structure

```text
fds_extract/
    FDS_Project-main/
        .gitignore
        DecisionTrees.ipynb
        KNNRegression.ipynb
        LinearRegression.ipynb
        MultiLogReg.ipynb
        NeuralNetworks.ipynb
        README.md
        StudentPerformanceAnalysis.ipynb
        report/
            .DS_Store
            DTheatmap1.png
            DTheatmap2.png
            DTnew.png
            DToversamp.png
            DTstructure.png
            FP0.png
            FP1.png
            figures/
                grades_errors.png
                grades_errors2.png
```

### Main Components

| Notebook | Description |
|---|---|
| `StudentPerformanceAnalysis.ipynb` | Exploratory analysis of the dataset and preliminary preprocessing |
| `LinearRegression.ipynb` | Linear regression experiments and evaluation |
| `KNNRegression.ipynb` | K-Nearest Neighbors regression analysis |
| `DecisionTrees.ipynb` | Decision tree models and performance comparison |
| `MultiLogReg.ipynb` | Multinomial logistic regression experiments |
| `NeuralNetworks.ipynb` | Neural network implementation and optimization |

---

## Objectives

The project explores:

- exploratory data analysis (EDA);
- preprocessing and feature engineering techniques;
- regression and classification workflows;
- comparison between classical ML models and neural approaches;
- evaluation of model performance using standard metrics.

---

## Technologies

The implementation is based on Python and Jupyter notebooks.

### Main Libraries

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `PyTorch`
- `imbalanced-learn`

---

## Methods

The repository includes experiments with several machine learning methods:

- Linear Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Multinomial Logistic Regression
- Feedforward Neural Networks

Additional topics covered include:

- feature scaling;
- oversampling techniques;
- hyperparameter tuning;
- model validation;
- visualization of results.

---

## Installation

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

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch imbalanced-learn notebook
```

---

## Usage

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open any notebook from the project directory.

Suggested reading order:

1. `StudentPerformanceAnalysis.ipynb`
2. `LinearRegression.ipynb`
3. `KNNRegression.ipynb`
4. `DecisionTrees.ipynb`
5. `MultiLogReg.ipynb`
6. `NeuralNetworks.ipynb`

---

## Notes

- This repository was developed for academic purposes.
- Some notebooks may rely on datasets or preprocessing steps defined in other notebooks.
- Figures and report material are stored inside the `report/` directory.

---

## Authors

Developed for the **Foundations of Data Science** course (Winter Semester 2023).

- Tommaso Leonardi
- Arianna Paolini
- Stefano Saravalle
- Paolo Cursi
- Pietro Signorino
