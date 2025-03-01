# Telecom Customer Churn Prediction

## Introduction

Customer churn refers to the phenomenon where customers stop using a company's products or services. In the highly competitive telecom industry, predicting customer churn is crucial for developing strategies to retain customers and maintain revenue. This project aims to build a predictive model to identify customers who are likely to churn.

## Dataset

The dataset used in this project contains information about customer demographics, services subscribed, account information, and whether the customer has churned. Key features include:

- **CustomerID**: Unique identifier for each customer
- **Gender**: Male or Female
- **SeniorCitizen**: Indicates if the customer is a senior citizen
- **Partner**: Whether the customer has a partner
- **Dependents**: Whether the customer has dependents
- **Tenure**: Number of months the customer has stayed with the company
- **PhoneService**: Whether the customer has phone service
- **MultipleLines**: Whether the customer has multiple lines
- **InternetService**: Type of internet service (DSL, Fiber optic, None)
- **OnlineSecurity**: Whether the customer has online security
- **OnlineBackup**: Whether the customer has online backup
- **DeviceProtection**: Whether the customer has device protection
- **TechSupport**: Whether the customer has tech support
- **StreamingTV**: Whether the customer has streaming TV
- **StreamingMovies**: Whether the customer has streaming movies
- **Contract**: Type of contract (Month-to-month, One year, Two year)
- **PaperlessBilling**: Whether the customer has paperless billing
- **PaymentMethod**: Payment method (Electronic check, Mailed check, Bank transfer, Credit card)
- **MonthlyCharges**: The amount charged to the customer monthly
- **TotalCharges**: The total amount charged to the customer
- **Churn**: Whether the customer has churned (Yes or No)

## Data Preprocessing

Data preprocessing steps include:

1. **Handling Missing Values**: Identifying and filling or removing missing data.
2. **Encoding Categorical Variables**: Converting categorical variables into numerical formats using techniques like one-hot encoding.
3. **Feature Scaling**: Normalizing numerical features to ensure all features contribute equally to the model.
4. **Splitting Data**: Dividing the dataset into training and testing sets to evaluate model performance.

## Exploratory Data Analysis (EDA)

EDA involves analyzing the dataset to discover patterns, relationships, and insights. Key steps include:

- **Distribution Analysis**: Understanding the distribution of numerical features.
- **Correlation Analysis**: Identifying relationships between features and the target variable.
- **Churn Rate Analysis**: Calculating the churn rate and analyzing factors contributing to churn.

## Model Building

Several machine learning models are employed to predict customer churn:

1. **Logistic Regression**: A linear model for binary classification problems.
2. **Decision Tree Classifier**: A non-linear model that splits data based on feature values.
3. **Random Forest Classifier**: An ensemble method that builds multiple decision trees and merges them for a more accurate prediction.
4. **Support Vector Machine (SVM)**: A model that finds the hyperplane that best separates the classes.

## Model Evaluation

Models are evaluated using metrics such as:

- **Accuracy**: The proportion of correctly predicted instances.
- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were identified correctly.
- **F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A table used to describe the performance of a classification model.

## Results

The performance of each model is compared, and the model with the best performance metrics is selected. Feature importance is analyzed to understand the factors contributing most to customer churn.

## Conclusion

The project successfully builds a predictive model for customer churn, providing insights into the key factors influencing churn. These insights can help telecom companies develop targeted strategies to improve customer retention.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Usage

To run the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/thississid/Telecom_Customer_Churn_Prediction.git

2. Navigate to the project directory:
   ```bash
   cd Telecom_Customer_Churn_Prediction

3. Install the required packages:
   ```bash
   pip install -r requirements.txt

4. Run the Flask script:

   ```bash
   cd FlaskApp
   python app.py

