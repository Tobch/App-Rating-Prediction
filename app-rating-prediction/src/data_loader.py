# Step 1: Project Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Data Loading
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Step 3: Data Exploration
print(train_data.info())
print(train_data.describe())
sns.histplot(train_data['Y'], bins=30)
plt.title('Distribution of App Ratings')
plt.show()

# Step 4: Data Preprocessing
# Handle missing values
train_data.fillna(method='ffill', inplace=True)

# Convert categorical variables to numerical
train_data = pd.get_dummies(train_data, columns=['X1', 'X6', 'X7'], drop_first=True)

# Split features and target
X = train_data.drop(columns=['Y'])
y = train_data['Y']

# Step 5: Model Selection
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training and Evaluation
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    mse = mean_squared_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    results[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}

# Step 7: Model Comparison
results_df = pd.DataFrame(results).T
results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Comparison')
plt.ylabel('Error Metrics')
plt.show()

# Step 8: Prediction on Test Data
# Preprocess test data similarly
test_data.fillna(method='ffill', inplace=True)
test_data = pd.get_dummies(test_data, columns=['X1', 'X6', 'X7'], drop_first=True)
test_predictions = models['Random Forest'].predict(test_data)

# Step 9: Results Visualization
# Assuming we have actual ratings for test data
# sns.scatterplot(x=test_data['actual_ratings'], y=test_predictions)
# plt.title('Predicted vs Actual Ratings')
# plt.show()

# Step 10: Documentation and Reporting
# Document findings and insights