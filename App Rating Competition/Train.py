import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from lazypredict.Supervised import LazyRegressor

# Load the data
train_data = pd.read_csv(r'D:\asu\Semester 6 - Spring 23\CSE472 - Artificial Intelligence\another version of Der3 version\App Rating Competition\cleaned_train.csv')
valid_data = pd.read_csv(r'D:\asu\Semester 6 - Spring 23\CSE472 - Artificial Intelligence\another version of Der3 version\App Rating Competition\after_split/val.csv')
test_data = pd.read_csv(r'D:\asu\Semester 6 - Spring 23\CSE472 - Artificial Intelligence\another version of Der3 version\App Rating Competition\after_split/test.csv')

# Feature columns (excluding the target 'Rating' and any identifier columns)
#feature_columns = ['NumReviews', 'AppSize', 'NumInstalls', 'IsFree', 'Price','Cat_AUTO_AND_VEHICLES', 'Cat_BOOKS_AND_REFERENCE', 'Cat_BUSINESS', 'Cat_COMMUNICATION', 'Cat_DATING', 'Cat_EDUCATION', 'Cat_ENTERTAINMENT', 'Cat_EVENTS', 'Cat_FAMILY', 'Cat_FINANCE', 'Cat_FOOD_AND_DRINK', 'Cat_GAME', 'Cat_HEALTH_AND_FITNESS', 'Cat_HOUSE_AND_HOME', 'Cat_LIBRARIES_AND_DEMO', 'Cat_LIFESTYLE', 'Cat_MAPS_AND_NAVIGATION', 'Cat_MEDICAL', 'Cat_NEWS_AND_MAGAZINES', 'Cat_OTHER', 'Cat_PERSONALIZATION', 'Cat_PHOTOGRAPHY', 'Cat_PRODUCTIVITY', 'Cat_SHOPPING', 'Cat_SOCIAL', 'Cat_SPORTS', 'Cat_TOOLS', 'Cat_TRAVEL_AND_LOCAL', 'Cat_VIDEO_PLAYERS', 'Cat_WEATHER', 'Age_Adults only 18+', 'Age_Everyone', 'Age_Everyone 10+', 'Age_Mature 17+', 'Age_Teen', 'Age_Unrated', 'Action', 'Action & Adventure', 'Adventure', 'Arcade', 'Art & Design', 'Auto & Vehicles', 'Beauty', 'Board', 'Books & Reference', 'Brain Games', 'Business', 'Card', 'Casino', 'Casual', 'Comics', 'Communication', 'Creativity', 'Dating', 'Education', 'Educational', 'Entertainment', 'Events', 'Finance', 'Food & Drink', 'Health & Fitness', 'House & Home', 'Libraries & Demo', 'Lifestyle', 'Maps & Navigation', 'Medical', 'Music', 'Music & Audio', 'Music & Video', 'News & Magazines', 'Parenting', 'Personalization', 'Photography', 'Pretend Play', 'Productivity', 'Puzzle', 'Racing', 'Role Playing', 'Shopping', 'Simulation', 'Social', 'Sports', 'Strategy', 'Tools', 'Travel & Local', 'Trivia', 'Video Players & Editors', 'Weather', 'Word', 'Year']
feature_columns = ['NumReviews', 'AppSize', 'NumInstalls', 'IsFree', 'Price', 'Cat_BUSINESS', 'Cat_COMMUNICATION',  'Cat_FAMILY', 'Cat_FINANCE', 'Cat_GAME', 'Cat_HEALTH_AND_FITNESS',  'Cat_LIFESTYLE', 'Cat_MEDICAL', 'Cat_OTHER', 'Cat_PERSONALIZATION', 'Cat_PHOTOGRAPHY', 'Cat_PRODUCTIVITY', 'Cat_SPORTS', 'Cat_TOOLS', 'Cat_TRAVEL_AND_LOCAL',  'Age_Everyone', 'Age_Everyone 10+', 'Age_Mature 17+', 'Age_Teen',  'Action', 'Action & Adventure', 'Adventure', 'Arcade', 'Art & Design', 'Auto & Vehicles', 'Beauty', 'Board', 'Books & Reference', 'Brain Games', 'Business', 'Card', 'Casino', 'Casual', 'Comics', 'Communication', 'Creativity', 'Dating', 'Education', 'Educational', 'Entertainment', 'Events', 'Finance', 'Food & Drink', 'Health & Fitness', 'House & Home', 'Libraries & Demo', 'Lifestyle', 'Maps & Navigation', 'Medical', 'Music', 'Music & Video', 'News & Magazines', 'Parenting', 'Personalization', 'Photography', 'Pretend Play', 'Productivity', 'Puzzle', 'Racing', 'Role Playing', 'Shopping', 'Simulation', 'Social', 'Sports', 'Strategy', 'Tools', 'Travel & Local', 'Trivia', 'Video Players & Editors', 'Weather', 'Word', 'Year']
X_train = train_data[feature_columns]
X_valid = valid_data[feature_columns]
X_test = test_data[feature_columns]

# Target column
y_train = train_data['Rating']
y_valid = valid_data['Rating']
y_test = test_data['Rating']

# Check for missing values in the target column
print(f"Missing values in y_train: {y_train.isnull().sum()}")

# Drop rows with missing target values
train_data = train_data.dropna(subset=['Rating'])
y_train = train_data['Rating']
X_train = train_data[feature_columns]

# Verify that there are no missing values
print(f"Missing values in y_train after handling: {y_train.isnull().sum()}")

# Check for missing values in y_valid
print(f"Missing values in y_valid: {y_valid.isnull().sum()}")

# Drop rows with missing target values in y_valid
valid_data = valid_data.dropna(subset=['Rating'])
y_valid = valid_data['Rating']
X_valid = valid_data[feature_columns]

# Check for missing values in X_valid
print(f"Missing values in X_valid: {X_valid.isnull().sum()}")

# Drop rows with missing values in X_valid
X_valid = X_valid.dropna()

# Ensure shapes match
print(f"Shape of X_valid: {X_valid.shape}")
print(f"Shape of y_valid: {y_valid.shape}")

# Check for missing values in y_test
print(f"Missing values in y_test: {y_test.isnull().sum()}")

# Drop rows with missing target values in y_test
test_data = test_data.dropna(subset=['Rating'])
y_test = test_data['Rating']
X_test = test_data[feature_columns]

# Check for missing values in X_test
print(f"Missing values in X_test: {X_test.isnull().sum()}")

# Drop rows with missing values in X_test
X_test = X_test.dropna()

# Ensure shapes match
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")

# === LazyML ===
print("\nRunning LazyML...")
lazy_regressor = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = lazy_regressor.fit(X_train, X_valid, y_train, y_valid)
print(models)

# === K-Fold Cross-Validation ===
print("\nRunning K-Fold Cross-Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define models to try
models = {
    'LinearRegression': LinearRegression(),
    'Lasso': Lasso(alpha=0.1),
    ##'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42,max_depth=None,min_samples_split=10,min_samples_leaf=2),
    'KNN': KNeighborsRegressor(    n_neighbors=100,
    weights='distance',           # Give closer neighbors more influence
    # algorithm='auto',             # Let sklearn choose the best algorithm
    # leaf_size=30,
    p=1,                          # Use Euclidean distance
    # metric='minkowski',
    n_jobs=-1  ),
    'SVR': SVR(kernel='rbf')
}

# Train, predict, evaluate, and save each model
for name, model in models.items():
    print(f"\nEvaluating model: {name}")
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    mean_cv_score = -cv_scores.mean()  # Convert negative MSE to positive
    print(f"{name} - Average CV MSE: {mean_cv_score:.4f}")

    # Train the model on the full training set
    model.fit(X_train, y_train)

    # Evaluate on validation and test sets
    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    valid_mse = mean_squared_error(y_valid, y_valid_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"{name} - Validation MSE: {valid_mse:.4f}")
    print(f"{name} - Test MSE: {test_mse:.4f}")

    # Save the model
    joblib.dump(model, f'{name}_model.joblib')
# gp -> Best Parameters: {'model_learning_rate': 0.05, 'modelmax_depth': 5, 'modelmin_samples_split': 2, 'modeln_estimators': 100, 'model_subsample': 0.8}
#knn -> Best Parameters: {'model_n_neighbors': 100, 'modelp': 1, 'model_weights': 'distance'}