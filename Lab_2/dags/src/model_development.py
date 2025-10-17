import os
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data from a CSV file
def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.

    Returns:
        bytes: Serialized data.
    """
    data = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/heart.csv"))
    return data

# Preprocess the data
def data_preprocessing(data):
    y = data['target']
    X = data.drop(columns=['target'])
    
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols),
        ],
        remainder='passthrough'  # keeps numeric columns as-is (no scaling needed for trees)
    )
    
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    return (X_train, X_test, y_train.values, y_test.values, preprocessor)

# Build and save a logistic regression model
def build_model(data, filename):
    X_train, X_test, y_train, y_test, _ = data
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Ensure the directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    
    # Save the trained model to a file
    with open(output_path, 'wb') as f:
        pickle.dump(rf, f)


# Load a saved logistic regression model and evaluate it
def load_model(data, filename):
    X_train, X_test, y_train, y_test, _ = data
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    
    with open(output_path, 'rb') as f:
        loaded_model = pickle.load(f)

    # Make predictions on the test data and print the model's score
    score = loaded_model.score(X_test, y_test)
    print(f"RandomForest accuracy on test data: {score:.4f}")

    predictions = loaded_model.predict(X_test)
    return predictions[0]


if __name__ == '__main__':
    x = load_data()
    prepped = data_preprocessing(x)
    build_model(prepped, 'rfmodel.sav')
    _ = load_model(prepped, 'rfmodel.sav')
