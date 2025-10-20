import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
# import seaborn as sns
import matplotlib.pyplot as plt

# Load the credit card fraud dataset
def load_data():
    data = pd.read_csv('samples/creditcard.csv')
    print("Dataset Shape:", data.shape)
    print("\nClass Distribution:")
    print(data['Class'].value_counts(normalize=True))
    return data

# Prepare the data
def prepare_data(data):
    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Calculate and display metrics
def evaluate_model(model, X_test, y_test, y_pred):
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ROC-AUC score
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC Score: {auc_roc:.4f}")
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():
    # Load data
    data = load_data()
    print(data.head(3))

    X = data.drop(columns=['Time', 'Amount', 'Class'])
    y = data['Class'].values
    
    print(f"X shape: {X.shape}, y shape: {y.shape}, fraud cases: {np.sum(y)}")

    model = LogisticRegression(class_weight={0:1, 1:2}, max_iter=1000)
    prediction = model.fit(X, y).predict(X)


    print(f"Model Score: {model.score(X, y):.4f}")
    print(f"X shape: {X.shape}, y shape: {y.shape}, fraud cases: {prediction.sum()}")

    grid = GridSearchCV(
            estimator=LogisticRegression(max_iter=1000),
            param_grid={'class_weight': [{0:1, 1:v} for v in np.linspace(1,20, 30)]},
            scoring= { 'precision' : make_scorer(precision_score), 'recall' : make_scorer(recall_score) },
            refit='precision',
            return_train_score=True,
            cv=10,
            n_jobs=-1
            )
    grid.fit(X, y)
    print("Best Parameters:", grid.best_params_)
    print(pd.DataFrame(grid.cv_results_))

    precision = precision_score(y, grid.predict(X))
    recall = recall_score(y, grid.predict(X))
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")


    plt.figure(figsize=(12,4))
    df = pd.DataFrame(grid.cv_results_)
    for score in ['mean_test_recall', 'mean_test_precision']:
        plt.plot([_[1] for _ in df['param_class_weight']], df[score], label=score)
    
    plt.legend()
    plt.show()



    # Prepare data
    # X_train, X_test, y_train, y_test = prepare_data(data)
    
    # TODO: Train your model here
    # Example:
    # from sklearn.ensemble import RandomForestClassifier
    # model = RandomForestClassifier(random_state=42)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    
    # TODO: Evaluate your model
    # evaluate_model(model, X_test, y_test, y_pred)

if __name__ == "__main__":
    main()
