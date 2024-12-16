import os
import numpy as np
import pandas as pd
from CokevsSal_Models import store_results, predict
from CNN import plt_hisory, rnn_model

def get_files(folder_path):

    meta_train_coke = pd.read_csv(os.path.join(folder_path,'COKE_TRAIN_metadata.csv'))
    meta_train_sal = pd.read_csv(os.path.join(folder_path,'SAL_TRAIN_metadata.csv'))

    meta_test_coke = pd.read_csv(os.path.join(folder_path,'COKE_TEST_metadata.csv'))
    meta_test_sal = pd.read_csv(os.path.join(folder_path,'SAL_TEST_metadata.csv'))

    X_train_coke = np.load(os.path.join(folder_path, 'COKE_TRAIN.npz'))
    X_train_sal = np.load(os.path.join(folder_path, 'SAL_TRAIN.npz'))

    X_test_coke = np.load(os.path.join(folder_path, 'COKE_TEST.npz'))
    X_test_sal = np.load(os.path.join(folder_path, 'SAL_TEST.npz'))

    y_train_coke = meta_train_coke['Day'].apply(lambda x: 0 if x == 'D1' else 1)
    y_train_sal = meta_train_sal['Day'].apply(lambda x: 0 if x == 'D1' else 1)


    y_test_coke = meta_test_coke['Day'].apply(lambda x: 0 if x == 'D1' else 1)
    y_test_sal = meta_test_sal['Day'].apply(lambda x: 0 if x == 'D1' else 1)

    print('X_train_coke shape :', X_train_coke['neuron_data'].shape)
    print('X_train_sal shape :', X_train_sal['neuron_data'].shape)

    print('X_test_coke shape :', X_test_coke['neuron_data'].shape)
    print('X_test_sal shape :', X_test_sal['neuron_data'].shape)

    print('y_train_coke shape :', y_train_coke.shape)
    print('y_train_sal shape :', y_train_sal.shape)

    print('y_test_coke shape :', y_test_coke.shape)
    print('y_test_sal shape :', y_test_sal.shape)

    X_train = np.vstack([X_train_coke['neuron_data'], X_train_sal['neuron_data']])
    X_test = np.vstack([X_test_coke['neuron_data'], X_test_sal['neuron_data']])

    y_train = np.hstack([y_train_coke, y_train_sal])
    y_test = np.hstack([y_test_coke, y_test_sal])

    print('X_train shape :', X_train.shape)
    print('X_test shape :', X_test.shape)

    print('y_train shape :', y_train.shape)
    print('y_test shape :', y_test.shape)

    return X_train, X_test, y_train, y_test, X_train_coke['neuron_data'], X_test_coke['neuron_data'], y_train_coke, y_test_coke, X_train_sal['neuron_data'], X_test_sal['neuron_data'], y_train_sal, y_test_sal

def train_model(X_train, y_train, X_test, y_test, day, label, folder_path):
    """
    Trains RNN models on the given dataset and evaluates their performance.

    This function processes the input data to extract features for training
    and testing RNN models for three different feature types: AUC, AMP, and FRQ.
    It trains a separate model for each feature type, evaluates its performance,
    and stores the results.

    Parameters:
    X_train (ndarray): Training data input features.
    y_train (ndarray): Training data labels.
    X_test (ndarray): Testing data input features.
    y_test (ndarray): Testing data labels.
    day (str): A string representing the day of the experiment.
    label (str): A string label for the model to be trained.
    folder_path (str): Path to the folder where model output will be saved.

    The function prints the shape of the training data, saves the training history
    plots, and stores the model performance metrics in a CSV file.
    """
    # Define the feature set
    feature_set = {
        'AUC': 0,
        'AMP': 1,
        'FRQ': 2
    }

    # Print the shape of the training data
    print(X_train.shape)

    # Train and evaluate the RNN model for each feature type
    for feature in feature_set:
        print(f'Training RNN model for feature type {feature}')
        
        # Train RNN model
        hist, best_model_path, last_model_path = rnn_model(
            X_train[:, :, feature_set[feature]].reshape(X_train.shape[0], X_train.shape[1], 1),
            X_test[:, :, feature_set[feature]].reshape(X_test.shape[0], X_test.shape[1], 1),
            y_train, y_test, os.path.join(folder_path, label, feature)
        )
        
        # Save training history plot
        plt_hisory(hist, os.path.join(folder_path, label, feature, f'RNN_{feature}.png'))

        # Predict and evaluate performance
        acc, pre, rec, f1 = predict(
            X_test[:, :, feature_set[feature]].reshape(X_test.shape[0], X_test.shape[1], 1),
            y_test, os.path.join(folder_path, label, feature, 'RNN_best_model_3.keras')
        )
        
        # Store the results
        store_results(
            'D1vsD7_Allday.csv', day, day + label, 'RNN', 'D1vsD7', feature, acc, pre, rec, f1
        )

    # Uncomment to train and evaluate using all features combined
    '''
    print('Training RNN model for combined features AUC_AMP_FRQ')
    hist, best_model_path, last_model_path = rnn_model(
        X_train, X_test, y_train, y_test, os.path.join(folder_path, label, 'AUC_AMP_FRQ')
    )
    plt_hisory(hist, os.path.join(folder_path, label, 'AUC_AMP_FRQ', 'RNN_AUC_AMP_FRQ.png'))

    acc, pre, rec, f1 = predict(
        X_test, y_test, os.path.join(folder_path, label, 'AUC_AMP_FRQ', 'RNN_best_model_3.keras')
    )
    store_results(
        'D1vsD7_Allday.csv', '15SEC', day + label, 'RNN', 'D1vsD7', 'AUC_AMP_FRQ', acc, pre, rec, f1
    )
    '''


if __name__ == "__main__":

    
    path_dict = {
            '15SEC':['15SEC/S1', '15SEC/S4'],
            '10SEC':['10SEC/S1', '10SEC/S4'],
            '5SEC':['5SEC/S1', '5SEC/S4']  
        }


    for day, folder_paths in path_dict.items():
        X_train, X_test, y_train, y_test, X_train_coke, X_test_coke, y_train_coke, y_test_coke, X_train_sal, X_test_sal, y_train_sal, y_test_sal = [], [], [], [], [], [], [], [], [], [], [], []
        for folder_path in folder_paths:
            X_train_f, X_test_f, y_train_f, y_test_f, X_train_coke_f, X_test_coke_f, y_train_coke_f, y_test_coke_f, X_train_sal_f, X_test_sal_f, y_train_sal_f, y_test_sal_f = get_files(folder_path)
            X_train.append(X_train_f)
            X_test.append(X_test_f)
            y_train.append(y_train_f)
            y_test.append(y_test_f)
            X_train_coke.append(X_train_coke_f)
            X_test_coke.append(X_test_coke_f)
            y_train_coke.append(y_train_coke_f)
            y_test_coke.append(y_test_coke_f)
            X_train_sal.append(X_train_sal_f)
            X_test_sal.append(X_test_sal_f)
            y_train_sal.append(y_train_sal_f)
            y_test_sal.append(y_test_sal_f)

        X_train = np.vstack(X_train)
        X_test = np.vstack(X_test)
        y_train = np.hstack(y_train)
        y_test = np.hstack(y_test)
        X_train_coke = np.vstack(X_train_coke)
        X_test_coke = np.vstack(X_test_coke)
        y_train_coke = np.hstack(y_train_coke)
        y_test_coke = np.hstack(y_test_coke)
        X_train_sal = np.vstack(X_train_sal)
        X_test_sal = np.vstack(X_test_sal)
        y_train_sal = np.hstack(y_train_sal)
        y_test_sal = np.hstack(y_test_sal)

        train_model(X_train, y_train, X_test, y_test, day, 'COKE&SAL', day)
        train_model(X_train_coke, y_train_coke, X_test_coke, y_test_coke, day, 'COKE', day)
        train_model(X_train_sal, y_train_sal, X_test_sal, y_test_sal, day, 'SAL', day)

    
    