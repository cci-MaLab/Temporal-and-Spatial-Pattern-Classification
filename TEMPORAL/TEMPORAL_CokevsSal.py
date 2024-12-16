from CNN import get_files, rnn_model, plt_hisory
import os
import tensorflow.keras as keras
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import csv
import numpy as np

def predict(X_test, y_test, model_path):
    """
    Load a trained model and make predictions on the test data.

    Args:
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True labels for the test data.
        model_path (str): Path to the trained model file.

    Returns:
        tuple: A tuple containing accuracy, precision, recall, and F1 score (macro average).
    """
    # Load the model
    model = keras.models.load_model(model_path)

    # Make predictions on the test data
    predictions = model.predict(X_test)
    # Convert probabilities to binary predictions
    predictions = (predictions > 0.5).astype(int)

    # Print the classification report
    print(classification_report(y_test, predictions))

    # Calculate accuracy, precision, recall, and F1 score (macro average)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    # Return the calculated metrics
    return accuracy, precision, recall, f1

def store_results(file_path, dataset_name, session, model_name, classification_type, feature_set, accuracy, precision, recall, f1):
    """Store the results of a model in a CSV file.

    Args:
        file_path (str): The path to the CSV file to store the results in.
        dataset_name (str): The name of the dataset used for training.
        session (str): The name of the session used for training.
        model_name (str): The name of the model used for training.
        classification_type (str): The type of classification used (e.g. 'COKE&SAL' or 'COKE').
        feature_set (str): The feature set used for training.
        accuracy (float): The accuracy of the model.
        precision (float): The precision of the model.
        recall (float): The recall of the model.
        f1 (float): The f1 score of the model.

    Returns:
        None
    """
    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode if it exists, else create a new one
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file doesn't exist, write the header first
        if not file_exists:
            writer.writerow(['Dataset', 'Session', 'Model', 'Classification', 'Feature Set', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

        # Write the results as a new row in the file
        writer.writerow([dataset_name, session, model_name, classification_type, feature_set, accuracy, precision, recall, f1])

def train_model(X_train, X_test, y_train, y_test, folder_path, dataset, session):
    """
    Train and evaluate RNN models for the given dataset and session.

    This function trains RNN models for the given dataset and session using the feature sets AUC, AMP, FRQ, and AUC_AMP_FRQ.
    It evaluates the performance of each model and stores the results in a CSV file.

    Parameters:
    X_train (ndarray): Training data input features.
    X_test (ndarray): Testing data input features.
    y_train (ndarray): Training data labels.
    y_test (ndarray): Testing data labels.
    folder_path (str): Path to the folder where model output will be saved.
    dataset (str): Name of the dataset.
    session (str): Name of the session.

    Returns:
    None
    """
    # List of feature sets
    feature_set = {
        'AUC': 0,
        'AMP': 1,
        'FRQ': 2
    }

    # Train and evaluate RNN models for each feature set
    for feature in feature_set:
        print(f'Training RNN model for feature type {feature}')
        
        # Train RNN model
        hist, best_model_path, last_model_path = rnn_model(
            X_train[:, :, feature_set[feature]].reshape(X_train.shape[0], X_train.shape[1], 1),
            X_test[:, :, feature_set[feature]].reshape(X_test.shape[0], X_test.shape[1], 1),
            y_train, y_test, os.path.join(folder_path, feature)
        )
        
        # Save training history plot
        plt_hisory(hist, os.path.join(folder_path, feature, f'RNN_{feature}.png'))
        
        # Predict and evaluate performance
        acc, pre, rec, f1 = predict(
            X_test[:, :, feature_set[feature]].reshape(X_test.shape[0], X_test.shape[1], 1),
            y_test, os.path.join(folder_path, feature, 'RNN_best_model_3.keras')
        )
        
        # Store results in a CSV file
        store_results(
            'Model_Performances_RNN.csv', dataset, session, 'RNN', 'CokevsSal', feature, acc, pre, rec, f1
        )

    # Train and evaluate RNN model for all feature sets
    '''hist, best_model_path, last_model_path = rnn_model(X_train, X_test, y_train, y_test, folder_path+'/AUC_AMP_FRQ')
    plt_hisory(hist, os.path.join(folder_path, 'AUC_AMP_FRQ', 'RNN_AUC_AMP_FRQ.png'))
    acc, pre, rec, f1 = predict(X_test, y_test, os.path.join(folder_path,'AUC_AMP_FRQ', 'RNN_best_model_3.keras'))
    store_results('Model_Performances_RNN.csv', dataset, session, 'RNN', 'CokevsSal', 'AUC_AMP_FRQ', acc, pre, rec, f1)'''

if __name__ == '__main__':

    """
    Train and evaluate RNN models for all sessions and datasets using the feature sets AUC, AMP, FRQ, and AUC_AMP_FRQ.
    """
    # List of datasets
    datasets = ['5SEC', '10SEC', '15SEC']

    path_dicts = [
        {'5SEC': '5SEC',
        'D1': '5SEC/D1',
        'D7': '5SEC/D7',
        'S1': '5SEC/S1',
        'S4': '5SEC/S4',
        'D1S1' :'5SEC/D1S1',
        'D1S4' : '5SEC/D1S4',
        'D7S1' : '5SEC/D7S1',
        'D7S4' : '5SEC/D7S4'},
        {'10SEC': '10SEC',
        'D1': '10SEC/D1',
        'D7': '10SEC/D7',
        'S1': '10SEC/S1',
        'S4': '10SEC/S4',
        'D1S1' :'10SEC/D1S1',
        'D1S4' : '10SEC/D1S4',
        'D7S1' : '10SEC/D7S1',
        'D7S4' : '10SEC/D7S4'},
        {'15SEC': '15SEC',
        'D1': '15SEC/D1',
        'D7': '15SEC/D7',
        'S1': '15SEC/S1',
        'S4': '15SEC/S4',
        'D1S1' :'15SEC/D1S1',
        'D1S4' : '15SEC/D1S4',
        'D7S1' : '15SEC/D7S1',
        'D7S4' : '15SEC/D7S4'}
        ]

    for path_dict, dataset in zip(path_dicts, datasets):
        for session, folder_path in path_dict.items():
            print(f'Training model for {session} in {dataset}')
            X_train, X_test, y_train, y_test = get_files(folder_path)
            train_model(X_train, X_test, y_train, y_test, session, dataset, folder_path)
            
            
    

   
    
