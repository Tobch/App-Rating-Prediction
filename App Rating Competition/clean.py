import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import argparse
import os

def clean_data(input_file, output_file, is_training=True):
    """
    Clean and preprocess app data for machine learning.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str
        Path to save the cleaned CSV file
    is_training : bool
        Whether the data is training data (with Rating column) or test data
    """
    print(f"Processing {input_file}...")
    
    # Load the data
    df = pd.read_csv(input_file)
    
    # Rename columns
    df.rename(columns={
        'X0': 'AppName',
        'X1': 'Category',
        'X2': 'NumReviews',
        'X3': 'AppSize',
        'X4': 'NumInstalls',
        'X5': 'IsFree',
        'X6': 'Price',
        'X7': 'AgeCategory',
        'X8': 'Genres',
        'X9': 'LastUpdate',
        'X10': 'Version',
        'X11': 'MinAndroidVer'
    }, inplace=True)
    
    if is_training:
        df.rename(columns={'Y': 'Rating'}, inplace=True)
    
    # Drop AppName
    df.drop(columns=['AppName'], inplace=True)
    
    # Fix Category Column
    # Remove the erroneous '1.9' category row
    df = df[df['Category'] != '1.9']
    
    # Group smaller categories
    category_counts = df['Category'].value_counts()
    small_categories = category_counts[category_counts < 50].index.tolist()
    df['Category_Grouped'] = df['Category'].apply(
        lambda x: 'OTHER' if x in small_categories else x)
    
    # One-hot encode the grouped categories
    category_dummies = pd.get_dummies(df['Category_Grouped'], prefix='Cat')
    df = pd.concat([df, category_dummies], axis=1)
    df.drop(columns=['Category', 'Category_Grouped'], inplace=True)
    
    # Fix NumReviews
    df["NumReviews"] = pd.to_numeric(df["NumReviews"], errors='raise')
    
    # Fix AppSize
    def convert_to_mb(size_str):
        if isinstance(size_str, str):
            if 'k' in size_str.lower():
                return float(size_str.lower().replace('k', '').strip()) / 1024
            elif 'm' in size_str.lower():
                return float(size_str.lower().replace('m', '').strip())
            elif 'varies with device' in size_str.lower():
                return np.nan
            else:
                return float(size_str)
        return size_str
    
    df["AppSize"] = df["AppSize"].apply(convert_to_mb)
    df["AppSize"] = pd.to_numeric(df["AppSize"], errors='coerce')
    app_size_median = df["AppSize"].median()
    df["AppSize"] = df["AppSize"].fillna(app_size_median)
    
    # Fix NumInstalls
    df["NumInstalls"] = df["NumInstalls"].map(
        lambda x: x[:-1].replace(',', '') if x.endswith('+') else x)
    df["NumInstalls"] = pd.to_numeric(df["NumInstalls"], errors='raise')
    
    # Fix IsFree and Price
    df["IsFree"] = df["IsFree"].map(lambda x: 0 if x == "Free" else 1)
    df["Price"] = pd.to_numeric(df["Price"], errors='raise')
    
    # Fix AgeCategory
    age_category_dummies = pd.get_dummies(df['AgeCategory'], prefix='Age')
    df = pd.concat([df, age_category_dummies], axis=1)
    df.drop('AgeCategory', axis=1, inplace=True)
    
    # Fix Genres
    df['Genre_split'] = df['Genres'].str.split(';')
    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(
        mlb.fit_transform(df['Genre_split']),
        columns=mlb.classes_,
        index=df.index
    )
    df = pd.concat([df, genre_encoded], axis=1)
    df.drop(['Genres', 'Genre_split'], axis=1, inplace=True)
    
    # Convert LastUpdate to Year
    df["LastUpdate"] = pd.to_datetime(df["LastUpdate"], format="%d-%b-%y")
    df["Year"] = df["LastUpdate"].dt.year
    df.drop(columns=["LastUpdate"], inplace=True)
    
    # Drop Version and MinAndroidVer columns
    df.drop(['Version', 'MinAndroidVer'], axis=1, inplace=True)
    
    # Convert boolean columns to int (0/1)
    bool_columns = df.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df[col] = df[col].astype(int)
    
    # Make sure all float columns are properly formatted
    float_columns = df.select_dtypes(include=['float64']).columns
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Make sure Year is properly formatted as numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('int32')
    
    # Handle missing values
    if is_training:
        # For training data, drop rows with missing Rating values
        df = df.dropna(subset=['Rating'])

    #----------Normalization
    df['NumInstalls'] = np.log1p(df['NumInstalls'])
    df['NumReviews'] = np.log1p(df['NumReviews']) 


    
    # Handle any other missing values with appropriate imputation
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        if col != 'Rating' or not is_training:
            df[col] = df[col].fillna(df[col].median())
    
    # Verify data quality
    try:
        assert df.isnull().sum().sum() == 0, "There are still NaN values in the DataFrame."
        assert df.select_dtypes(include=['object']).empty, "There are still text columns in the DataFrame."
        assert not np.isinf(df).any().any(), "There are infinite values in the DataFrame."
        print("Data validation passed!")
    except AssertionError as e:
        print(f"Warning: {e}")
    
    # Save the cleaned data
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved to {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Clean app data for machine learning')
    parser.add_argument('--train', required=True, help='Path to training data CSV')
    parser.add_argument('--test', required=True, help='Path to test data CSV')
    parser.add_argument('--output-dir', default='./cleaned', help='Directory to save cleaned data')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process training data
    train_output = os.path.join(args.output_dir, 'cleaned_train.csv')
    clean_data(args.train, train_output, is_training=True)
    
    # Process test data
    test_output = os.path.join(args.output_dir, 'cleaned_test.csv')
    clean_data(args.test, test_output, is_training=False)
    
    print("Data cleaning complete!")

if __name__ == "__main__":
    main()

    # python clean_data.py \
    #   --train /Users/ziad/Desktop/ML/der3/app-rating-regression-ML-/train.csv \
    #   --test /Users/ziad/Desktop/ML/der3/app-rating-regression-ML-/test.csv \
    #   --output-dir /Users/ziad/Desktop/ML/der3/app-rating-regression-ML-/cleaned_data
