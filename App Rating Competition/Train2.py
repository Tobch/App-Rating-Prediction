import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score

# Load the full dataset
data = pd.read_csv(r'D:\asu\Semester 6 - Spring 23\CSE472 - Artificial Intelligence\another version of Der3 version\Der3 version/normalized_train.csv')

# Feature columns (same as before)
feature_columns = ['NumReviews', 'AppSize', 'NumInstalls', 'IsFree', 'Price', 'Cat_BUSINESS', 'Cat_COMMUNICATION',  'Cat_FAMILY', 'Cat_FINANCE', 'Cat_GAME', 'Cat_HEALTH_AND_FITNESS',  'Cat_LIFESTYLE', 'Cat_MEDICAL', 'Cat_OTHER', 'Cat_PERSONALIZATION', 'Cat_PHOTOGRAPHY', 'Cat_PRODUCTIVITY', 'Cat_SPORTS', 'Cat_TOOLS', 'Cat_TRAVEL_AND_LOCAL',  'Age_Everyone', 'Age_Everyone 10+', 'Age_Mature 17+', 'Age_Teen',  'Action', 'Action & Adventure', 'Adventure', 'Arcade', 'Art & Design', 'Auto & Vehicles', 'Beauty', 'Board', 'Books & Reference', 'Brain Games', 'Business', 'Card', 'Casino', 'Casual', 'Comics', 'Communication', 'Creativity', 'Dating', 'Education', 'Educational', 'Entertainment', 'Events', 'Finance', 'Food & Drink', 'Health & Fitness', 'House & Home', 'Libraries & Demo', 'Lifestyle', 'Maps & Navigation', 'Medical', 'Music', 'Music & Video', 'News & Magazines', 'Parenting', 'Personalization', 'Photography', 'Pretend Play', 'Productivity', 'Puzzle', 'Racing', 'Role Playing', 'Shopping', 'Simulation', 'Social', 'Sports', 'Strategy', 'Tools', 'Travel & Local', 'Trivia', 'Video Players & Editors', 'Weather', 'Word', 'Year']  # use your existing feature_columns list

X = data[feature_columns]
y = data['Rating']

# Drop rows with missing target values
data = data.dropna(subset=['Rating'])
y = data['Rating']
X = data[feature_columns]

# K-Fold Cross-Validation (optional, for evaluation)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=None, min_samples_split=10, min_samples_leaf=2),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42, max_depth=5, min_samples_split=2, subsample=0.8),
    'KNN': KNeighborsRegressor(n_neighbors=100, weights='distance', p=1, n_jobs=-1),
    'SVR': SVR(kernel='rbf')
}

for name, model in models.items():
    print(f"\nTraining model: {name}")
    # Optional: cross-validation
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    print(f"{name} - Average CV MSE: {-cv_scores.mean():.4f}")

    # Train on all data
    model.fit(X, y)
    joblib.dump(model, f'{name}_model.joblib')
    print(f"{name} model saved.")

print("Training complete.")