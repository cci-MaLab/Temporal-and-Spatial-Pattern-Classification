import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import csv
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

import warnings
warnings.filterwarnings('ignore')


def XGB_clf(X_train, y_train, X_test, y_test):
    """
    XGBoost classifier.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    X_test : array-like
        Testing features.
    y_test : array-like
        Testing labels.

    Returns
    -------
    accuracy : float
        Accuracy score.
    precision : float
        Precision score.
    recall : float
        Recall score.
    f1 : float
        F1 score.
    """
    print("Training XGBOOST model ....")
    # Create an XGBClassifier
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Create a StratifiedKFold object
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define the parameter grid
    params = {
        'max_depth': [3, 4, 5],  # Maximum depth of the tree
        'learning_rate': [0.01, 0.1, 0.2],  # Learning rate
        'n_estimators': [100, 200],  # Number of trees
        'subsample': [0.8, 1],  # Subsample ratio
        'colsample_bytree': [0.8, 1]  # Column sample ratio
    }

    print("Fitting GridSearchCV...")
    # Setup GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=params, scoring='accuracy', cv=kfold, verbose=1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    print("Best parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    print("Making predictions on the test set...")
    predictions = best_model.predict(X_test)

    # Calculate the accuracy, precision, recall, and f1 score
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    # Return the calculated metrics
    print("Returning calculated metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1}")
    return accuracy, precision, recall, f1

def SVM_clf(X_train, y_train, X_test, y_test):
    """
    Trains a Support Vector Machine (SVM) classifier with GridSearchCV.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    X_test : array-like
        Testing features.
    y_test : array-like
        Testing labels.

    Returns
    -------
    accuracy : float
        Accuracy score.
    precision : float
        Precision score.
    recall : float
        Recall score.
    f1 : float
        F1 score.
    """
    print("Training SVM model...")
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'kernel': ['linear', 'rbf', 'poly'],  # Type of SVM
        'degree': [2, 3, 4]  # Degree of the polynomial kernel function (ignored by all other kernels)
    }

    # Create a SVM Classifier
    svm = SVC()

    # Create a StratifiedKFold object
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Setup GridSearchCV
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=kfold, scoring='accuracy')

    # Fit GridSearchCV
    print("Fitting GridSearchCV...")
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    print("Best parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    print("Making predictions on the test set...")
    predictions = best_model.predict(X_test)
    
    # Calculate the accuracy, precision, recall, and f1 score
    print("Calculating accuracy, precision, recall, and f1 score...")
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    # Return the calculated metrics
    print("Returning calculated metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1}")
    return accuracy, precision, recall, f1


def create_model(input_dim, l2_factor=0.01):
    """
    Creates a neural network model with the given input dimension and L2 regularization factor.

    The model consists of four dense layers with dropout layers in between. The first and second dense layers have 128 and 64 units, respectively, and use the ReLU activation function. The third layer has 64 units and uses the ReLU activation function as well. The output layer has one unit and uses the sigmoid activation function.

    :param input_dim: The input dimension of the model.
    :type input_dim: int
    :param l2_factor: The L2 regularization factor, defaults to 0.01.
    :type l2_factor: float, optional
    :return: The created neural network model.
    :rtype: Sequential
    """
    print(f"Creating model with input_dim={input_dim} and l2_factor={l2_factor}")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(l2_factor)),
        Dropout(0.5),  # Dropout after the first dense layer
        Dense(64, activation='relu', kernel_regularizer=l2(l2_factor)),
        Dropout(0.5),  # Another dropout after the second dense layer
        Dense(64, activation='relu', kernel_regularizer=l2(l2_factor)),  # Continue with the additional layer
        Dropout(0.5),  # Dropout also after the additional layer
        Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_factor))
    ])
    print(f"Model created: {model}")
    return model


def Neural_model(X_train, y_train, X_test, y_test):
    """
    Trains a neural network model on the given data and evaluates its performance on the test set.

    :param X_train: The feature array for the training set.
    :type X_train: numpy.ndarray
    :param y_train: The label array for the training set.
    :type y_train: numpy.ndarray
    :param X_test: The feature array for the test set.
    :type X_test: numpy.ndarray
    :param y_test: The label array for the test set.
    :type y_test: numpy.ndarray
    :return: The accuracy, precision, recall, and f1 score of the model.
    :rtype: tuple
    """

    print("Creating model with input dimension:", X_train.shape[1])
    # Create a neural network model with the given input dimension
    model = create_model(X_train.shape[1])

    print("Compiling model with Adam optimizer and binary cross-entropy loss")
    # Compile the model with the Adam optimizer and binary cross-entropy loss
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("Training model on the training set...")
    # Train the model on the training set
    history = model.fit(X_train, y_train,
                        epochs=10, batch_size=32, verbose=1)

    print("Evaluating model on the test set...")
    # Evaluate the model on the test set
    score = model.evaluate(X_test, y_test, verbose=0)

    print("Making predictions on the test set...")
    # Make predictions on the test set
    predictions = model.predict(X_test, verbose=0)
    # Convert probabilities to binary predictions
    predictions = (predictions > 0.5).astype(int)

    print("Calculating accuracy, precision, recall, and f1 score...")
    # Calculate the accuracy, precision, recall, and f1 score
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    print("Returning calculated metrics...")
    # Return the calculated metrics
    return accuracy, precision, recall, f1


def store_results(file_path, dataset_name, session, model_name, classification_type, feature_set, accuracy, precision, recall, f1):
    """
    Stores the results of a model evaluation in a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file where the results will be stored.
    dataset_name : str
        Name of the dataset.
    session : str
        Session name.
    model_name : str
        Name of the model.
    classification_type : str
        Type of classification (e.g. binary, multi-class).
    feature_set : str
        Set of features used for the evaluation.
    accuracy : float
        Accuracy score.
    precision : float
        Precision score.
    recall : float
        Recall score.
    f1 : float
        F1 score.
    """
    print(f"Storing results in {file_path}")
    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode if it exists, else create a new one
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file doesn't exist, write the header first
        if not file_exists:
            print("Writing header to file")
            writer.writerow(['Dataset', 'Session', 'Model', 'Classification', 'Feature Set', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

        print("Writing results to file")
        # Write the results as a new row in the file
        writer.writerow([dataset_name, session,  model_name,  classification_type, feature_set, accuracy, precision, recall, f1])


def process_day_session(day_value, session_value, labels_train, labels_test, amp_train, amp_test, auc_train, auc_test, frq_train, frq_test, 
                amp_train_prev, amp_test_prev, auc_train_prev, auc_test_prev, frq_train_prev, frq_test_prev, 
                y_train, y_test,dataset, combination, classify,feature_set=['AUC', 'AMP', 'FRQ', 'AUC_AMP_FRQ'], models=['XGBOOST', 'SVM', 'Neural Network']):
    """
    This function processes the data for a given day (0 or 1) by filtering the train/test sets, selecting the feature set (AMP, AUC, FRQ),
    and training different models (XGBOOST, SVM, Neural Network) with and without Alp_Prev.
    
    Parameters:
    - day_value (int): The day value to filter by (0 or 1)
    - labels_train, labels_test: Arrays with labels and session information
    - amp_train, amp_test, auc_train, auc_test, frq_train, frq_test: Feature arrays
    - amp_train_prev, amp_test_prev, auc_train_prev, auc_test_prev, frq_train_prev, frq_test_prev: Previous feature arrays
    - y_train, y_test: Labels for train and test
    - feature_set (list): List of features to process (AMP, AUC, FRQ)
    - models (list): List of models to train (XGBOOST, SVM, Neural Network)
    - dataset, combination (str): Strings for tracking dataset and combination info
    
    Returns:
    - None (outputs plots and performance metrics)
    """
    print("Processing day", day_value, "and session", session_value)
    # Filter by the day value
    if day_value is not None and session_value is not None:
        condition_train = (labels_train[:, 3] == day_value) & (labels_train[:, 4] == session_value)
        condition_test = (labels_test[:, 3] == day_value) & (labels_test[:, 4] == session_value)
    elif day_value is not None:
        condition_train = (labels_train[:, 3] == day_value)
        condition_test = (labels_test[:, 3] == day_value)
    elif session_value is not None:
        condition_train = (labels_train[:, 4] == session_value)
        condition_test = (labels_test[:, 4] == session_value)
    else:
        condition_train = slice(None)
        condition_test = slice(None)


    # Apply the condition to filter the data
    amp_train_filtered = amp_train[condition_train]
    amp_test_filtered = amp_test[condition_test]
    auc_train_filtered = auc_train[condition_train]
    auc_test_filtered = auc_test[condition_test]
    frq_train_filtered = frq_train[condition_train]
    frq_test_filtered = frq_test[condition_test]

    amp_train_prev_filtered = amp_train_prev[condition_train]
    amp_test_prev_filtered = amp_test_prev[condition_test]
    auc_train_prev_filtered = auc_train_prev[condition_train]
    auc_test_prev_filtered = auc_test_prev[condition_test]
    frq_train_prev_filtered = frq_train_prev[condition_train]
    frq_test_prev_filtered = frq_test_prev[condition_test]

    y_train_filtered = y_train[condition_train]
    y_test_filtered = y_test[condition_test]

    if day_value == 0:
        day_str = 'D1'
    else:
        day_str = 'D7'

    # Loop over the selected feature set and train models
    for feat in feature_set:
        print(feat)
        # Feature selection
        if feat == "AMP":
            
            train = amp_train_filtered
            test = amp_test_filtered
            train_prev = amp_train_prev_filtered
            test_prev = amp_test_prev_filtered

        elif feat == "AUC":
            continue
            train = auc_train_filtered
            test = auc_test_filtered
            train_prev = auc_train_prev_filtered
            test_prev = auc_test_prev_filtered

        elif feat == "FRQ":
            continue
            train = frq_train_filtered
            test = frq_test_filtered
            train_prev = frq_train_prev_filtered
            test_prev = frq_test_prev_filtered

        elif feat == 'AUC_AMP_FRQ':
            continue
            train = np.concatenate((auc_train_filtered, amp_train_filtered, frq_train_filtered), axis=1)
            test = np.concatenate((auc_test_filtered, amp_test_filtered, frq_test_filtered), axis=1)
            train_prev = np.concatenate((auc_train_prev_filtered, amp_train_prev_filtered, frq_train_prev_filtered), axis=1)
            test_prev = np.concatenate((auc_test_prev_filtered, amp_test_prev_filtered, frq_test_prev_filtered), axis=1)

        # Train with and without Alp_Prev for each model
        for model_name in models:
            # Train and evaluate the model without Alp_Prev
            if model_name == 'XGBOOST':
                continue
                print("Training XGBOOST")
                acc, pre, rec, f1 = XGB_clf(train, y_train_filtered, test, y_test_filtered)
                store_results('Classification_Performance.csv', dataset, combination, 'XGBOOST', classify, feat, acc, pre, rec, f1)
            elif model_name == 'SVM':
                print("Training SVM")
                acc, pre, rec, f1 = SVM_clf(train, y_train_filtered, test, y_test_filtered)
                store_results('CvsS_Rem.csv', dataset, combination, 'SVM', classify, feat, acc, pre, rec, f1)
                
            elif model_name == 'Neural Network':
                continue
                print("Training Neural Network")
                acc, pre, rec, f1 = Neural_model(train, y_train_filtered, test, y_test_filtered)
                store_results('Classification_Performance.csv', dataset, combination, 'Neural_Network', classify, feat, acc, pre, rec, f1)

            # Save the results 
            # append_to_csv(dataset, combination, feat, "Without Alp_Prev", model_name, acc, pre, rec, f1)

            # Train and evaluate the model with Alp_Prev
            '''if model_name == 'XGBOOST':
                acc, pre, rec, f1 = XGB_clf(train_prev, y_train_filtered, test_prev, y_test_filtered)
            elif model_name == 'SVM':
                acc, pre, rec, f1 = SVM_clf(train_prev, y_train_filtered, test_prev, y_test_filtered)
            elif model_name == 'Neural Network':
                acc, pre, rec, f1 = Neural_model(train_prev, y_train_filtered, test_prev, y_test_filtered)'''
def train_models(dataset, pca_file, classify):

    print("Loading data from", pca_file)
    pca = np.load(pca_file)

    amp_train = pca['amp_train']
    amp_test = pca['amp_test']

    auc_train = pca['auc_train']
    auc_test = pca['auc_test']

    frq_train = pca['frq_train']
    frq_test = pca['frq_test']

    files_train = pca['files_train']
    files_test = pca['files_test']
    
    labels_train = pca['y_train']
    labels_test = pca['y_test']

    alp_prev_train = labels_train[:,1].reshape(-1,1)
    alp_prev_test = labels_test[:,1].reshape(-1,1)

    alp_train = labels_train[:,0].reshape(-1,1)
    alp_test = labels_test[:,0].reshape(-1,1)

    amp_train_prev = np.concatenate([amp_train, alp_train, alp_prev_train], axis =1)
    auc_train_prev = np.concatenate([auc_train, alp_train, alp_prev_train], axis =1)
    frq_train_prev = np.concatenate([frq_train, alp_train, alp_prev_train], axis =1)

    amp_test_prev = np.concatenate([amp_test, alp_test, alp_prev_test], axis =1)
    auc_test_prev = np.concatenate([auc_test, alp_test, alp_prev_test], axis =1)
    frq_test_prev = np.concatenate([frq_test, alp_test, alp_prev_test], axis =1)

    y_train = labels_train[:,2]
    y_test = labels_test[:,2]

    print("Processing day 0 and session 0 - D1S1")
    process_day_session(0, 0, labels_train, labels_test, amp_train, amp_test, auc_train, auc_test, frq_train, frq_test, 
                amp_train_prev, amp_test_prev, auc_train_prev, auc_test_prev, frq_train_prev, frq_test_prev, 
                y_train, y_test, dataset, 'D1S1', classify)

    print("Processing day 0 and session 1 - D1S4")
    process_day_session(0, 1, labels_train, labels_test, amp_train, amp_test, auc_train, auc_test, frq_train, frq_test, 
                amp_train_prev, amp_test_prev, auc_train_prev, auc_test_prev, frq_train_prev, frq_test_prev, 
                y_train, y_test, dataset, 'D1S4', classify)
    
    print("Processing day 1 and session 0 - D7S1")
    process_day_session(1, 0, labels_train, labels_test, amp_train, amp_test, auc_train, auc_test, frq_train, frq_test, 
                amp_train_prev, amp_test_prev, auc_train_prev, auc_test_prev, frq_train_prev, frq_test_prev, 
                y_train, y_test, dataset, 'D7S1', classify)

    print("Processing day 1 and session 1 - D7S4")
    process_day_session(1, 1, labels_train, labels_test, amp_train, amp_test, auc_train, auc_test, frq_train, frq_test, 
                amp_train_prev, amp_test_prev, auc_train_prev, auc_test_prev, frq_train_prev, frq_test_prev, 
                y_train, y_test, dataset, 'D7S4', classify)

    print("Processing day 1 and session None - D7")
    process_day_session(1, None, labels_train, labels_test, amp_train, amp_test, auc_train, auc_test, frq_train, frq_test, 
                amp_train_prev, amp_test_prev, auc_train_prev, auc_test_prev, frq_train_prev, frq_test_prev, 
                y_train, y_test, dataset, 'D7', classify)
    
    print("Processing day None and session None")
    process_day_session(None, None, labels_train, labels_test, amp_train, amp_test, auc_train, auc_test, frq_train, frq_test, 
                amp_train_prev, amp_test_prev, auc_train_prev, auc_test_prev, frq_train_prev, frq_test_prev, 
                y_train, y_test, dataset, '15SEC', classify)
    
    print("Processing day 0 and session None - D1")
    process_day_session(0, None, labels_train, labels_test, amp_train, amp_test, auc_train, auc_test, frq_train, frq_test, 
                amp_train_prev, amp_test_prev, auc_train_prev, auc_test_prev, frq_train_prev, frq_test_prev, 
                y_train, y_test, dataset, 'D1', classify)
    
    
    


if __name__ == '__main__':

    dataset = '5SEC'
    classify = 'CokeVsSal'
    pca_file = '5SEC/5SEC_coke_sal_pca.npz'
    train_models(dataset, pca_file, classify)

        

