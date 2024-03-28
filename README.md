# credit-risk-classification

## Background
The aim of this project was to train and evaulate a logistic regression model using historical lending activity data to identify the creditworthiness of borrowers.

The dataset included information on:
    * loan size
    * interest rate
    * borrower's income
    * debt to income ratio
    * number of accounts the borrower has held
    * derogatory marks against the borrower
    * total debt

## Methods
First, the lending data (77,536 rows or borrowers) were imported from a CSV file into a Pandas Dataframe in the Jupyter Notebook (`credit_risk_classification.ipynb`). The data were then split into training and testing datasets using the sklearn library's `train_test_split` function.

We fit a logistic regression model using the training data. Logistic regression was deemed most appropriate for this analysis because we were interested in the loan status, a binary variable indicated by 0 as healthy and 1 as high risk of defaulting, as the dependent variable. We saved the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.

Lastly, we evaluated the model's performance by generaing a confusion matrix and printing out the classification report.

## Results
The following was our confusion matrix:
```
[[18679    80]
 [   67   558]]
```
The classification report:
```
                precision    recall  f1-score   support

           0       1.00      1.00      1.00     18759
           1       0.87      0.89      0.88       625

    accuracy                           0.99     19384
   macro avg       0.94      0.94      0.94     19384
weighted avg       0.99      0.99      0.99     19384
```
* Our model predicted well with 99% accuracy overall
* It predicted with 100% precision, recall and F1-score for the healthy loan group
* The precision, recall and F1-score metrics fell to 87%, 89% and 88%, respectively, for the higher risk group

## Conclusion
We recommend our model to predict loans but with some caveats. It does a great job in predicting both the healthy and high-risk loan groups with 99% accuracy. However, the precision score fell from 100% for the healthy loan borrowers to 87% for the high-risk borrowers. This is an example of accuracy being susceptible to imbalanced classes--the number of good loans substantially outweighs the number of at-risk loans. The model may do a fair job of predicting good loans, but its performance wanes, as indicated by lower precision, recall and F1-score, when it comes to identifying higher-risk loan candidates.