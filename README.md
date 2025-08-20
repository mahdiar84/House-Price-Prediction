🏠 House Price Prediction
📌 Project Overview

This project aims to predict house prices using the famous Kaggle House Prices Dataset
.
We explore multiple regression models, preprocess categorical and numerical features, and apply GridSearchCV for hyperparameter tuning to find the best model.

📂 Dataset

Source: Kaggle – House Prices: Advanced Regression Techniques

Target variable: SalePrice

Features: Includes numerical (e.g., LotArea, YearBuilt, GrLivArea) and categorical features (e.g., Neighborhood, HouseStyle).

⚙️ Project Workflow

Data Cleaning:

Dropped irrelevant columns (Id).

Filled missing values (mean for numerical, most frequent for categorical).

Feature Engineering:

OneHotEncoding for categorical features.

StandardScaler for numerical features.

Model Training:
Models tested:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Hyperparameter Tuning:

Used GridSearchCV to optimize parameters for each model.

Evaluation Metrics:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

📊 Results

Each model was trained and tuned.

Random Forest and Gradient Boosting gave the most accurate predictions compared to Linear Regression and Decision Tree.

Visualizations include:

Error comparison bar charts

Predicted vs Actual prices scatter plot

📌 Future Work

Experiment with XGBoost, LightGBM, CatBoost.

Feature selection techniques.

Use ensemble stacking for better accuracy.

📈 Example Visualizations

Confusion matrix not applicable (regression), but:

Error comparison bar chart.

Predicted vs Actual price scatter plot.

📝 Author

Developed by Mahdiar Naghizadeh
With ideas and guidance refined through AI collaboration.
