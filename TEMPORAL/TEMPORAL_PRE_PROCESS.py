import numpy as np
import pandas as pd
import os

def create_neuron_dataframe(folder_path, output_file, target_time_steps=180):
    """
    Create a neuron data array from AUC, AMP, and FRQ data files.

    Parameters:
    folder_path (str): Path to the folder containing the AUC, AMP, and FRQ data files.
    output_file (str): Path to the output file for the neuron data array.
    target_time_steps (int): Target number of time steps for each neuron's data.

    Returns:
    metadata_df (pd.DataFrame): A DataFrame containing metadata for each neuron.
    """
    file_names = os.listdir(os.path.join(folder_path, 'AMP'))

    neuron_data = []
    metadata = []

    for file_name in file_names:
        print(f"Processing {file_name}")
        
        # Extract mouse, day, and session from the file name
        try:
            mouse, day, session = file_name[:-4].split('_')
        except ValueError:
            # Handle different naming conventions
            parts = file_name[:-4].split('_')
            if len(parts) == 2:
                mouse = parts[0]
                day, session = parts[1][0:2], parts[1][2:]
            elif len(parts) > 3:
                mouse, day, session = parts[0], parts[1], '_'.join(parts[2:])
            else:
                raise ValueError(f"Unexpected file name format: {file_name}")

        # Load the corresponding file from each directory
        amp_df = pd.read_csv(os.path.join(folder_path, 'AMP', file_name))
        auc_df = pd.read_csv(os.path.join(folder_path, 'AUC', file_name))
        frq_df = pd.read_csv(os.path.join(folder_path, 'FRQ', file_name))

        # Check the number of time steps
        num_time_steps = amp_df.shape[0]
        if target_time_steps is None:
            target_time_steps = num_time_steps  # Set target time steps based on the first file

        # Select all neurons' data (assuming no other columns besides neurons and ALP)
        amp = amp_df.iloc[:, :-3].values  # Get all neurons' AMP data
        auc = auc_df.iloc[:, :-3].values  # Get all neurons' AUC data
        frq = frq_df.iloc[:, :-3].values  # Get all neurons' FRQ data
        neuron_ids = amp_df.columns[:-3]  # Get neuron IDs (column names)

        # Pad sequences if they have fewer time steps than the target
        if num_time_steps < target_time_steps:
            padding = np.zeros((target_time_steps - num_time_steps, amp.shape[1]))
            amp = np.vstack((amp, padding))
            auc = np.vstack((auc, padding))
            frq = np.vstack((frq, padding))

        # Process ALP column (assumed to be the same for all neurons in the AMP file)
        # alp = amp_df['ALP'].apply(lambda x: 0 if x < 1 else 1 if x == 1 else x).values
        # alp = np.reshape(alp, (1, num_time_steps))  # Reshape to (1, num_time_steps)
        # alp = np.tile(alp, (amp.shape[1], 1))  # Broadcast ALP to match shape (num_neurons, num_time_steps)
        
        # Pad ALP if necessary
        # if num_time_steps < target_time_steps:
        #     alp_padding = np.zeros((amp.shape[1], target_time_steps - num_time_steps))
        #     alp = np.hstack((alp, alp_padding))

        # Stack the features for each neuron and store metadata
        for i in range(amp.shape[1]):  # Iterate over each neuron
            #neuron_instance = np.stack([auc[:, i], amp[:, i], frq[:, i], alp[i, :]], axis=-1)
            neuron_instance = np.stack([auc[:, i], amp[:, i], frq[:, i]], axis = -1)
            #neuron_instance = frq[:,i].reshape(-1,1)
            neuron_data.append(neuron_instance)
            metadata.append({
                'Mouse': mouse,
                'Day': day,
                'Session': session,
                'Neuron_ID': neuron_ids[i]
            })

    # Convert the neuron data and metadata to a DataFrame and save it
    neuron_data_array = np.array(neuron_data)
    metadata_df = pd.DataFrame(metadata)
    print(neuron_data_array.shape)
    # Save the neuron data as a .npz file
    np.savez(output_file, neuron_data=neuron_data_array)
    
    # Return or save the metadata DataFrame
    metadata_df.to_csv(output_file.replace('.npz', '_metadata.csv'), index=False)

    print(f"Saved {len(neuron_data)} neuron instances to {output_file}")
    print(f"Saved metadata to {output_file.replace('.npz', '_metadata.csv')}")

    return metadata_df

if __name__ == "__main__":
    folder_paths = ['5SEC', '5SEC/D1', '5SEC/D1S1', '5SEC/D1S4', '5SEC/D7', '5SEC/D7S1', '5SEC/D7S4', '10SEC', 
                    '10SEC/D1', '10SEC/D1S1', '10SEC/D1S4', '10SEC/D7', '10SEC/D7S1', '10SEC/D7S4', 
                    '15SEC', '15SEC/D1', '15SEC/D1S1', '15SEC/D1S4', '15SEC/D7', '15SEC/D7S1', '15SEC/D7S4']

    for folder_path in folder_paths:

        metadata_df = create_neuron_dataframe(os.path.join(folder_path, 'COKE', 'TRAIN'), os.path.join(folder_path, 'COKE_TRAIN.npz'))
        metadata_df = create_neuron_dataframe(os.path.join(folder_path, 'SAL', 'TRAIN'), os.path.join(folder_path, 'SAL_TRAIN.npz'))

        metadata_df = create_neuron_dataframe(os.path.join(folder_path, 'COKE', 'TEST'), os.path.join(folder_path, 'COKE_TEST.npz'))
        metadata_df = create_neuron_dataframe(os.path.join(folder_path, 'SAL', 'TEST'), os.path.join(folder_path, 'SAL_TEST.npz'))
