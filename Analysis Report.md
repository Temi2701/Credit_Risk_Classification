# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

## Table of Contents 
* Overview of the Analysis
* Results
* Summary

# Overview

Lending companies lend money or assets to borrowers with the expectation that the borrower will either return the asset or repay the lender. Credit risk is associated with a borrower not returning an asset or paying a loan back, causing a lender to incur losses. This is typically measured by lenders in various ways. However, in this analysis, we will use machine learning to examine a dataset of historical lending activity from a peer-to-peer lending services company. The goal is to create a model that can identify the creditworthiness of borrowers.

Using a machine learning model, I aim to determine which loans are considered healthy (low-risk) or non-healthy (high-risk) based on the loan status provided by the lending company.

The Logistic Regression Algorithm is selected as the most suitable tool for our machine learning model since it is widely used to predict the probability of a target variable in classification problems. Utilizing the dataset provided by the lending company, I developed a Logistic Regression Model that achieved an accuracy score of 94%. However, it's important to note that the model's recall value (0.94) for non-healthy loans is lower than the recall value (0.99) for healthy loans. This indicates that the model is better at predicting loan statuses as healthy rather than accurately predicting non-healthy loan statuses. This discrepancy is attributed to the dataset being imbalanced, meaning that the majority of the data belongs to one class label (in this case, healthy loans significantly outweigh non-healthy loans).

Step 3 [Split the Data into Training and Testing Sets], using the value_counts function, we are able to see that the data is highly imbalanced. The majority class is healthy loans [0] and the minority class is non-healthy loans [1]:

code
lending_data_df['loan_status'].value_counts()

output
<img width="272" alt="valuecount" src="https://github.com/KajK0121/credit-risk-classification/assets/140313204/9cebc60d-31f1-4c82-b221-5b4879c1273b">

The stages of the machine learning process I went through as part of this analysis:
1. Split the dataset into features (X) and labels (y).
2. Segment the data into training and testing sets.
3. Construct a logistic regression model using both the original and resampled data.
4. Make predictions using the logistic regression model with both the original and resampled training data.
5. Evaluate the model's performance using accuracy scores, confusion matrices, and classification reports for both the original and resampled datasets.
6. Analyse observations derived from the evaluation reports.

   
Leveraging logistic regression models, these approaches collectively tackle the issue of imbalanced data through the utilisation of resampling techniques. Resampling guarantees a fairer representation of both classes during the training phase, thereby fostering balanced predictions and potentially enhancing the model's capability to detect high-risk loans. The selection of logistic regression as the modeling algorithm strikes a balance between interpretability and predictive strength.


## Results


### Machine Learning Model 1:

- **Accuracy:**
  - Overall Accuracy: 0.99 (exceptionally high)

- **Precision:**
  - Healthy Loans (0): 1.00
  - Non-Healthy Loans (1): 0.84 (slightly lower)

- **Recall:**
  - Healthy Loans: 0.99
  - Non-Healthy Loans: 0.94

- **Summary:**
  - The model excels in overall accuracy, achieving an exceptionally high score of 0.99.
  - Precision for healthy loans is perfect (1.00), showcasing strong accuracy in positive predictions.
  - Precision for non-healthy loans is slightly lower at 0.84, indicating the presence of some false positives.
  - The model demonstrates robust recall for both healthy (0.99) and non-healthy (0.94) loans, capturing the majority of instances effectively.
  - In summary, the logistic regression model showcases a high level of correctness in predictions for both healthy and non-healthy loans, with a balance between precision and recall metrics.


### Machine Learning Model 2:
- **Accuracy:**
  - Overall Accuracy: 0.99 (very high)

- **Precision:**
  - Healthy Loans (0): 1.00 (very high)
  - Non-Healthy Loans (1): 0.84 (solid accuracy)

- **Recall:**
  - Healthy Loans: 0.99 (strong ability to identify healthy instances)
  - Non-Healthy Loans: 0.99 (successful capture of a significant portion of non-healthy instances)

- **Summary:**
  - Model 2 exhibits exceptional accuracy, achieving a high score of 0.99.
  - Precision for healthy loans is very high (1.00), indicating minimal incorrect positive predictions.
  - Precision for non-healthy loans is solid at 0.84, showcasing accuracy in identifying true positive instances.
  - Robust recall for both healthy (0.99) and non-healthy (0.99) loans, demonstrating the model's effectiveness in capturing relevant instances.
  - In summary, Model 2 is highly effective in accurately classifying both healthy and non-healthy loans, with a strong balance between precision and recall metrics.

## Summary

In summary, both machine learning models did well in telling us if a loan is healthy or not. 

Model 1, trained on the original data, was really good at correctly saying a loan is healthy (precision), but it might miss a few unhealthy ones (recall). This model was super accurate overall, but it leans a bit more towards being careful not to wrongly say a loan is unhealthy.

Model 2, trained on balanced data, found a nice balance between being accurate about healthy loans and correctly identifying unhealthy ones. It was accurate overall, and it's good at both being careful about wrong predictions and catching most unhealthy loans.

Choosing between the two depends on what's more important for the specific problem. If it's critical not to wrongly label a loan as unhealthy, maybe go with Model 1. But if you want a good balance between being right about healthy loans and catching unhealthy ones, Model 2 could be the better choice.

In the end, both models are strong, and the decision depends on the specific goals and priorities of the problem you're working on.
