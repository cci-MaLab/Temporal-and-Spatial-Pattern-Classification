import numpy as np
import os
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import tensorflow.keras as keras
import time
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warning messages
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set GPU to use
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory to grow as needed

def get_files(folder_path):

    X_train_coke = np.load(os.path.join(folder_path, 'COKE_TRAIN.npz'))
    X_train_sal = np.load(os.path.join(folder_path, 'SAL_TRAIN.npz'))

    y_train_coke = np.ones(X_train_coke['neuron_data'].shape[0])
    y_train_sal = np.zeros(X_train_sal['neuron_data'].shape[0])

    X_test_coke = np.load(os.path.join(folder_path, 'COKE_TEST.npz'))
    X_test_sal = np.load(os.path.join(folder_path, 'SAL_TEST.npz'))

    y_test_coke = np.ones(X_test_coke['neuron_data'].shape[0])
    y_test_sal = np.zeros(X_test_sal['neuron_data'].shape[0])

    print('X_train_coke shape :', X_train_coke['neuron_data'].shape)
    print('X_train_sal shape :', X_train_sal['neuron_data'].shape)

    print('X_test_coke shape :', X_test_coke['neuron_data'].shape)
    print('X_test_sal shape :', X_test_sal['neuron_data'].shape)

    X_train = np.vstack([X_train_coke['neuron_data'], X_train_sal['neuron_data']])
    X_test = np.vstack([X_test_coke['neuron_data'], X_test_sal['neuron_data']])

    y_train = np.hstack([y_train_coke, y_train_sal])
    y_test = np.hstack([y_test_coke, y_test_sal])

    print('X_train shape :', X_train.shape)
    print('X_test shape :', X_test.shape)

    print('y_train shape :', y_train.shape)
    print('y_test shape :', y_test.shape)

    return X_train, X_test, y_train, y_test

def predict(model_path, X_test, y_test):

    model = keras.models.load_model(model_path)

    predictions  = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)
    print(classification_report(y_test, predictions))

def plt_hisory(hist, figname):
    """
    Plot the loss and accuracy over epochs.

    Parameters
    ----------
    hist : keras.callbacks.History
        The history of the model's performance over epochs.
    figname : str
        The name of the file to save the figure to.

    Returns
    -------
    None
    """
    # Get the index (epoch) of the best model based on validation loss
    best_epoch = np.argmin(hist.history['val_loss']) + 1  # Adding 1 because epochs start at 1

    # Get the best validation loss value and accuracy
    best_val_loss = hist.history['val_loss'][best_epoch - 1]  # Subtract 1 because Python indexing starts at 0
    best_val_accuracy = hist.history['val_accuracy'][best_epoch - 1]

    print(f"The best model was saved at epoch {best_epoch} with a validation loss of {best_val_loss:.4f} "
          f"and accuracy of {best_val_accuracy:.4f}.")

    # Plot the loss and accuracy over epochs
    plt.figure(figsize=(12, 6))

    # Plot the training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['loss'], label='Training Loss')
    plt.plot(hist.history['val_loss'], label='Validation Loss')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.scatter(best_epoch-1, best_val_loss, color='red')  # Mark the best validation loss
    plt.text(best_epoch, best_val_loss, f'{best_val_loss:.4f}', color='red', fontsize=12, ha='right')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['accuracy'], label='Training Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
    plt.axvline(x=best_epoch-1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.scatter(best_epoch-1, best_val_accuracy, color='red')  # Mark the best validation accuracy
    plt.text(best_epoch, best_val_accuracy, f'{best_val_accuracy:.4f}', color='red', fontsize=12, ha='right')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.savefig(figname)
    # Display the plots
    plt.tight_layout()
    #plt.show()

def cnn_model(X_train, X_test, y_train, y_test, folder_path):
    # Model Implementation
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=X_train.shape[1:]))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Use 'sigmoid' for binary classification

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    best_model_path = os.path.join(folder_path,'CNN1_best_model.keras')
    # Callback to save the best model
    model_checkpoint = ModelCheckpoint(filepath=best_model_path, monitor='val_loss', 
                                    save_best_only=True, verbose=1)

    # Train the model
    hist = model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(X_test, y_test), callbacks = [model_checkpoint])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.4f}')

    last_model_path = os.path.join(folder_path,'CNN1_last_model.keras')
    model.save(last_model_path)
    predict(last_model_path, X_test, y_test)

    return hist, best_model_path, last_model_path

def cnn_model_2(X_train, X_test, y_train, y_test, folder_path):
    # Model Implementation with Regularization
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=X_train.shape[1:]))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))  # Dropout for regularization
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))  # L2 regularization
    model.add(Dropout(0.5))  # Additional Dropout
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    best_model_path = os.path.join(folder_path, 'CNN2_best_model.keras')
    
    # Callbacks
    model_checkpoint = ModelCheckpoint(filepath=best_model_path, monitor='val_loss', 
                                       save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    
    # Train the model
    hist = model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=16,  # Increased batch size
        validation_data=(X_test, y_test), 
        callbacks=[model_checkpoint]
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Detailed classification report
    predict(best_model_path, X_test, y_test)
    
    last_model_path = os.path.join(folder_path, 'CNN2_last_model.keras')
    model.save(last_model_path)
    
    return hist, best_model_path, last_model_path

def rnn_model(X_train, X_test, y_train, y_test, folder_path):
    
    n_outputs = 1  # Assuming binary classification
    # Define the time-series input layer
    print(X_train.shape[1:])
    recurrent_input = Input(shape=X_train.shape[1:], name="TIMESERIES_INPUT")

    # RNN Layers

    # Layer 1: Bidirectional LSTM
    rec_layer_one = Bidirectional(LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), return_sequences=True), name="BIDIRECTIONAL_LAYER_1")(recurrent_input)
    rec_layer_one = Dropout(0.1, name="DROPOUT_LAYER_1")(rec_layer_one)

    # Layer 2: Bidirectional LSTM
    rec_layer_two = Bidirectional(LSTM(16, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), return_sequences=True), name="BIDIRECTIONAL_LAYER_2")(rec_layer_one)
    rec_layer_two = Dropout(0.1, name="DROPOUT_LAYER_2")(rec_layer_two)

    # Layer 3: Bidirectional LSTM
    rec_layer_three = Bidirectional(LSTM(8, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)), name="BIDIRECTIONAL_LAYER_3")(rec_layer_two)
    rec_layer_three = Dropout(0.1, name="DROPOUT_LAYER_3")(rec_layer_three)

    # Dense Layers (after RNN)
    combined_dense_two = Dense(4, activation='relu', name="DENSE_LAYER_2")(rec_layer_three)

    # Output layer (for binary classification)
    output = Dense(n_outputs, activation='sigmoid', name="OUTPUT_LAYER")(combined_dense_two)

    # Compile the model
    model = Model(inputs=[recurrent_input], outputs=[output])

    # Compile with binary cross-entropy loss
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print the model summary
    print(model.summary())

    best_model_path = os.path.join(folder_path, 'RNN_best_model_3.keras')

    model_checkpoint = ModelCheckpoint(filepath=best_model_path, monitor='val_loss', 
                                       save_best_only=True, verbose=1)
    
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    hist = model.fit(
            X_train, y_train, 
            epochs=100, 
            batch_size=16,  # Increased batch size
            validation_data=(X_test, y_test),
            callbacks=[model_checkpoint]
        )
    
    last_model_path = os.path.join(folder_path, 'RNN_last_model_3.keras')
    model.save(last_model_path)
    predict(last_model_path, X_test, y_test)
    
    return hist, best_model_path, last_model_path


if __name__ == '__main__':

    # Load data
    folder_path = '5SEC/D1'
    print("___________________________LOADING DATA____________________________")
    X_train, X_test, y_train, y_test = get_files(folder_path)
    
    print("_______________________________CNN 1___________________________________")
    hist, best_model_path, last_model_path = cnn_model(X_train, X_test, y_train.astype('int'), y_test.astype('int'), folder_path)
    plt_hisory(hist, os.path.join(folder_path, "CNN_1.png"))
    print("________________________BEST MODEL CNN 1___________________________")
    predict(best_model_path, X_test, y_test)

    print("_______________________________CNN 2___________________________________")
    hist, best_model_path, last_model_path = cnn_model_2(X_train, X_test, y_train.astype('int'), y_test.astype('int'), folder_path)
    plt_hisory(hist, os.path.join(folder_path,"CNN_2.png"))
    print("________________________BEST MODEL CNN 2___________________________")
    predict(best_model_path, X_test, y_test)

    print("_______________________________RNN___________________________________")
    hist, best_model_path, last_model_path = rnn_model(X_train, X_test, y_train.astype('int'), y_test.astype('int'), folder_path)
    plt_hisory(hist, os.path.join(folder_path,"RNN_3.png"))
    print("________________________BEST MODEL RNN___________________________")
    predict(best_model_path, X_test, y_test)