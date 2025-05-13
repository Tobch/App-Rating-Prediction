import pandas as pd
import joblib

# === Load the test data ===
test_data = pd.read_csv(r'D:\asu\Semester 6 - Spring 23\CSE472 - Artificial Intelligence\another version of Der3 version\App Rating Competition\normalized_test.csv')

# === Define the features (must match the training features) ===
feature_columns = ['NumReviews', 'AppSize', 'NumInstalls', 'IsFree', 'Price', 'Cat_BUSINESS', 'Cat_COMMUNICATION'
                   ,  'Cat_FAMILY', 'Cat_FINANCE', 'Cat_GAME', 'Cat_HEALTH_AND_FITNESS',  'Cat_LIFESTYLE', 'Cat_MEDICAL', 'Cat_OTHER', 'Cat_PERSONALIZATION', 'Cat_PHOTOGRAPHY', 'Cat_PRODUCTIVITY', 'Cat_SPORTS', 'Cat_TOOLS', 'Cat_TRAVEL_AND_LOCAL',  'Age_Everyone', 'Age_Everyone 10+', 'Age_Mature 17+', 'Age_Teen',  'Action', 'Action & Adventure', 'Adventure', 'Arcade', 'Art & Design', 'Auto & Vehicles', 'Beauty', 'Board', 'Books & Reference', 'Brain Games', 'Business', 'Card', 'Casino', 'Casual', 'Comics', 'Communication', 'Creativity', 'Dating', 'Education', 'Educational', 'Entertainment', 'Events', 'Finance', 'Food & Drink', 'Health & Fitness', 'House & Home', 'Libraries & Demo', 'Lifestyle', 'Maps & Navigation', 'Medical', 'Music', 'Music & Video', 'News & Magazines', 'Parenting', 'Personalization', 'Photography', 'Pretend Play', 'Productivity', 'Puzzle', 'Racing', 'Role Playing', 'Shopping', 'Simulation', 'Social', 'Sports', 'Strategy', 'Tools', 'Travel & Local', 'Trivia', 'Video Players & Editors', 'Weather', 'Word', 'Year']
X_test = test_data[feature_columns]
X_test = X_test.fillna(0)
# === Load your trained model ===
model = joblib.load(r'D:\asu\Semester 6 - Spring 23\CSE472 - Artificial Intelligence\another version of Der3 version\App Rating Competition\SVR_model.joblib')  # or 'RandomForest_model.joblib', etc.

# === Make predictions ===
test_data['Y'] = model.predict(X_test)

# === Save to a new CSV file ===
test_data.to_csv('SampleSubmission.csv', index=False)

print("Predictions saved to 'SampleSubmission.csv'")