#  Loan Repayment Prediction Using Ensemble Learning Methods

##  Objective

This project aims to build a machine learning model to predict whether a loan applicant is likely to repay a loan. It uses various ensemble learning techniques to improve predictive accuracy and assist banks in making data-driven lending decisions.

###  Dataset Overview

The dataset includes features such as:
- Interest rate (`int.rate`)
- Credit policy compliance (`credit.policy`)
- FICO score (`fico`)
- Number of recent credit inquiries (`inq.last.6mths`)
- Loan purpose (categorical)

### Preprocessing Steps
- **Null Values:** Verified to contain zero missing entries.
- **Categorical Encoding:** Label encoding was applied to the `purpose` column with six unique categories.
- **Train-Test Split:** Dataset was split for model training and evaluation.


###  Exploratory Data Analysis

Data visualizations revealed that the following features significantly influence the target variable (`not.fully.paid`):
- Interest rate
- Credit policy
- FICO score
- Recent inquiries

These insights guided model selection and evaluation.


###  Machine Learning Models Used

| Model                            | Accuracy         | Observations |
|----------------------------------|------------------|--------------|
| Decision Tree                    | 84.58%           | Baseline model |
| Bagging (with Decision Tree)     | 73.10% (mean)    | Lower than base |
| AdaBoost (with Decision Tree)    | 84%              | Comparable to base |
| Random Forest                    | **84.7%**        | Best performing |
| Gradient Boosting                | ~84%             | Similar to Random Forest |

> Ensemble methods generally performed well, with Random Forest achieving the highest accuracy.


###  Tech Stack

- Python
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for visualization
- Scikit-learn for machine learning models
- Google Colab for development


###  Conclusion
The project shows how ensemble methods can improve loan repayment prediction models. Among all techniques evaluated, **Random Forest Classifier** offered the best performance with 84.7% accuracy. However, other ensemble models like AdaBoost and Gradient Boosting also delivered similar results, confirming the robustness of the approach.


###  How to Run
1. Clone the repository or open the notebook directly in Google Colab:
   [Loan_Repayment_Prediction.ipynb](https://colab.research.google.com/github/shsarv/ML-and-its-Application/blob/main/Loan_Repayment_Prediction.ipynb)
2. Install necessary Python libraries (if not already installed):
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run all cells sequentially.


