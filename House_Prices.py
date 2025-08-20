import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv(r"C:\Users\saraye tel\OneDrive\Desktop\ARCH_Roadmap\Datasets\House_Prices\train.csv")
df = pd.DataFrame(data)

# Drop ID column if exists
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Fill missing values
df = df.fillna(df.mean(numeric_only=True))

# Separate target
y = df["SalePrice"]
X = df.drop("SalePrice", axis=1)

# Identify categorical & numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessor: scale numerical, encode categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Models and their parameter grids
models_and_params = {
    "RandomForest": (
        RandomForestRegressor(random_state=42),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
        }
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5]
        }
    ),
    "DecisionTree": (
        DecisionTreeRegressor(random_state=42),
        {
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5]
        }
    ),
    "LinearRegression": (
        LinearRegression(),
        {}  # No hyperparameters to tune
    )
}

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Results storage
results = {}

# Run GridSearchCV for each model
for name, (model, params) in models_and_params.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    grid = GridSearchCV(pipeline, param_grid=params, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # Best model
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    results[name] = {
        "Best Params": grid.best_params_,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
    
    print(f"Best Params: {grid.best_params_}")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")

# Compare results visually
scores = {name: res["R2"] for name, res in results.items()}
plt.figure(figsize=(8, 5))
sns.barplot(x=list(scores.keys()), y=list(scores.values()), palette="viridis")
plt.title("R² Score Comparison Across Models")
plt.ylabel("R²")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("R_2 png", dpi=300)
plt.show()