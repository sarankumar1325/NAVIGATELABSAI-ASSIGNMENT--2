# CART Algorithm: Classification Tree Implementation

## Table of Contents
- [Overview](#overview)
- [Introduction to CART Algorithm](#introduction-to-cart-algorithm)
- [Key Concepts and Terminology](#key-concepts-and-terminology)
- [Steps in Building the Classification Tree](#steps-in-building-the-classification-tree)
- [Advantages and Limitations](#advantages-and-limitations)
- [Practical Applications](#practical-applications)
- [Code Implementation](#code-implementation)
- [Results](#results)
- [Conclusion](#conclusion)

## Overview
This project demonstrates the implementation of a CART (Classification and Regression Trees) algorithm, specifically a classification tree model for predicting job offers based on candidate attributes. The code includes data preprocessing, model training, hyperparameter tuning, evaluation, and visualization of the decision tree.

## Introduction to CART Algorithm
The CART algorithm is a popular supervised learning method for both classification and regression tasks. In classification, it splits data into subsets based on attribute values, forming a decision tree:
- **Root Node**: Starting point of the tree.
- **Internal Nodes**: Represent decision points on attributes.
- **Branches**: Indicate possible outcomes of each decision.
- **Leaf Nodes**: Represent the final class labels or predictions.

## Key Concepts and Terminology
- **Entropy and Gini Impurity**: Metrics for evaluating splits.
- **Pruning**: Process to prevent overfitting by reducing tree complexity.
- **Hyperparameters**: Attributes like `max_depth`, `min_samples_split`, etc., that affect model performance.

## Steps in Building the Classification Tree
1. **Importing Libraries**: Install and import required libraries like `pydotplus`, `pandas`, and `scikit-learn`.
2. **Defining the Dataset**: A sample dataset with attributes such as CGPA, communication skills, practical knowledge, etc., is used.
3. **Data Preprocessing**:
   - Convert categorical data into numerical codes.
   - Split the data into training and test sets.
4. **Model Training**:
   - Use `GridSearchCV` to optimize hyperparameters.
   - Fit the model with the best parameters.
5. **Evaluation**:
   - Predict on test data and compute accuracy.
   - Generate a classification report.
6. **Visualization**: Use `export_graphviz` and `pydotplus` to visualize the decision tree.

## Advantages and Limitations
### Advantages
- **Interpretability**: Trees are easy to interpret and visualize.
- **Flexibility**: Can handle both categorical and numerical data.

### Limitations
- **Overfitting**: Can lead to complex trees that generalize poorly.
- **Instability**: Small changes in data can lead to different trees.

## Practical Applications
- **Job Prediction**: Estimate job offers based on candidate attributes.
- **Medical Diagnosis**: Predict diseases based on patient data.
- **Credit Scoring**: Assess loan eligibility and risk.

## Code Implementation

### Libraries Used
- `pandas`: For data manipulation and preparation.
- `sklearn.tree`: For building and visualizing the decision tree.
- `pydotplus`: For generating tree visualizations.
- `matplotlib`: For displaying the decision tree.

### Dataset
A dataset representing attributes such as CGPA, interactivity, and practical knowledge to predict job offers.

## Results
The classification tree achieved an accuracy of 100% on the test set, indicating perfect classification for the given data. This accuracy suggests the model may be overfitted to the small dataset used.

## Conclusion
This CART implementation highlights the power of decision trees for classification tasks. The model was able to accurately predict job offers based on the defined attributes. Further testing on a larger dataset is recommended to assess generalization.

## Authors
- Ajinesh.D - 22am007
- Sarankumar.S - 22am055
- Gobika.R - 22am069
---

⭐ **Star this repository if you found it helpful!**
