import os
import glob
import scipy.io
import pandas as pd
import numpy as np

header = ['Time','I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def read_and_save_mat_files(start_path,headers):
    # Traverse the directory structure
    counter =0
    for root, dirs, files in os.walk(start_path):
        # Use glob to find all .mat files in the current directory
        for file in glob.glob(os.path.join(root, '*.mat')):
            # Read the .mat file
            if counter == 1:
                break
            mat_data = scipy.io.loadmat(file)
            
            # Assuming the data you want to save is in the first variable of the .mat file
            # Adjust this as necessary based on the structure of your .mat files
            for key in mat_data:
                if not key.startswith('__'): # Ignore special keys
                    data = mat_data[key]
                    # Generate time data based on the total duration and the number of samples
                    total_duration = 10 # Total duration in seconds
                    sample_rate = 1000 # Sample rate in Hz (assuming 1000 Hz for this example)
                    time = np.linspace(0, total_duration, len(data[0]), endpoint=False)
                    
                 
                   # Add the time data as a new row to the existing data
                    data = np.vstack((time, data))
                    
                    # Create a DataFrame from the combined data
                    df = pd.DataFrame(data)
                    
                    # Transpose the DataFrame
                    df_transposed = df.T
                    
                    # Extract the record identifier from the file path
                    record_identifier = os.path.splitext(os.path.basename(file))[0]
                         # Write the headers to the CSV file
                    csv_file_path = os.path.join(root, f'{record_identifier}.csv')
                    
                    # Write the headers to the CSV file
                    with open(csv_file_path, 'w') as f:
                        f.write(','.join(headers) + '\n')
                    
                    # Append the data to the CSV file
                    df_transposed.to_csv(csv_file_path, mode='a', index=False, header=False)
            counter+=1
# Specify the root directory of your dataset
root_directory = './dataset/WFDBRecords'


# Call the function to read and save all .mat files
read_and_save_mat_files(root_directory, header)