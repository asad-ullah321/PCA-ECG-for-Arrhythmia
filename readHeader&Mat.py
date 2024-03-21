import psycopg2
import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt
import biosppy.signals.ecg as ecg
import pywt

dataset_path='.\dataset\dataset\WFDBRecords\\'
lowcut = 0.66
highcut = 46.0
def data_Insertor():
    i = 0
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        dbname="arrhythmia_dataset_wavelet",
        user="postgres",
        password="axdw1234",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    
    try:
        # createQuery = '''CREATE TABLE IF NOT EXISTS ECG(
        #         ID varchar(10) primary key,
        #         AGE int,
        #         GENDER varchar(10),
        #         DX TEXT,
        #         HR REAL,
        #         p_signal BYTEA,
        #         n_signal BYTEA)'''

        #for wavelet db
        createQuery = '''CREATE TABLE IF NOT EXISTS ECG(
                ID varchar(10) primary key,
                AGE int,
                GENDER varchar(10),
                DX TEXT,
                HR REAL,
                denoised_signal BYTEA)'''
                
                # QRS REAL,
                # NGraph BYTEA,
                # OGraph BYTEA,
        cursor.execute(createQuery)
            
        for dir1 in os.listdir(dataset_path):
            dir1_path = os.path.join(dataset_path, dir1)
            
            # Loop through the subdirectories
            for dir2 in os.listdir(dir1_path):
                dir2_path = os.path.join(dir1_path, dir2)
                
                # Loop through the ECG records
                for file_name in os.listdir(dir2_path):
                    if file_name.endswith('.mat'):
                        print(i," ",file_name)
                        file_path = os.path.join(dir2_path, file_name[:-4])
                        try:
                            record = wfdb.rdrecord(file_path)
                            write_to_db(conn, cursor, record)
                            i+=1
                        except Exception as e:
                            print(f"Error reading file {file_name}: {e}")
                            continue
                    
        # cursor.execute(f"SELECT id,p_signal FROM ECG where id ='JS00001' ")
        
        # row = cursor.fetchall()[0]
            
        # print(row[0])    
        #     # Convert fetched data to numpy array
        # # for row in rows:
        # data = np.frombuffer(row[1],dtype=float)
        # data = data.reshape((5000,12)) 
        # print(data[0])
                    
            

    
    finally:
        # Close database connection
        cursor.close()
        conn.close()


def write_to_db(conn, cursor, record):

    try:
        age = int(record.comments[0][5:])
    except:
        age = 62
                        
    # o_plot_data = visualize_data(record.p_signal)
    # print(record.p_signal[0])    
    # filtered_p_signal = np.zeros_like(record.p_signal)
    # for lead in range(12):
        # filtered_signal = bandpass_filter(record.p_signal[:, lead], lowcut, highcut, 500)
        # filtered_p_signal[:, lead] = filtered_signal
    filtered_p_signal =  wavelet_denoising(record.p_signal) 
    if record.record_name == 'JS00001':
        print(filtered_p_signal[0][:10])                 
    # n_plot_data = visualize_data(filtered_p_signal)
    print(filtered_p_signal.shape)                                   
    # insertValues = (record.record_name,
    #                 age,
    #                 record.comments[1][5:],
    #                 record.comments[2][4:],
    #                 calculate_heartrate(filtered_p_signal,record.fs),
    #                 # calculate_qrs_interval(filtered_p_signal,record.fs),
    #                 # psycopg2.Binary(n_plot_data),
    #                 # psycopg2.Binary(o_plot_data),
    #                 psycopg2.Binary(record.p_signal.tobytes()),
    #                 psycopg2.Binary(filtered_p_signal.tobytes()))
    # cursor.execute('''INSERT INTO ECG (ID, AGE, GENDER,DX,HR,p_signal,n_signal) 
    #                VALUES (%s,%s,%s,%s,%s,%s,%s)''',insertValues)

    ## for wavelet 
    insertValues = (record.record_name,
                    age,
                    record.comments[1][5:],
                    record.comments[2][4:],
                    calculate_heartrate(filtered_p_signal,record.fs),
                    # calculate_qrs_interval(filtered_p_signal,record.fs),
                    # psycopg2.Binary(n_plot_data),
                    # psycopg2.Binary(o_plot_data),
                    
                    psycopg2.Binary(filtered_p_signal.tobytes()))
    cursor.execute('''INSERT INTO ECG (ID, AGE, GENDER,DX,HR,denoised_signal) 
                   VALUES (%s,%s,%s,%s,%s,%s)''',insertValues)
    
    
    conn.commit()

# Define a Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    # Calculate the Nyquist frequency
    nyquist_freq = 0.5 * fs
    # Normalize the cutoff frequencies
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    # Design the Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

# Apply the bandpass filter to the input data
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    # Design the Butterworth bandpass filter
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Filter the input data using the designed filter
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def wavelet_denoising(data):
    # Apply wavelet denoising to each feature (column) of the data
    denoised_data = np.empty_like(data)
    for i in range(data.shape[1]):
        # Apply wavelet denoising to each feature
        coeffs = pywt.wavedec(data[:, i], 'db8', level=2) # Use Daubechies wavelet 'db8'
        # Set threshold to remove noise
        threshold = np.sqrt(2 * np.log(len(data)))
        coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
        # Reconstruct the signal
        denoised_data[:, i] = pywt.waverec(coeffs, 'db8')
    
    return denoised_data

def calculate_heartrate(record,fs):
    ecg_signal = record[:,1]
    # Process ECG signal to detect R-peaks
    rpeaks = ecg.engzee_segmenter(ecg_signal, sampling_rate=500)[0]
    # Calculate RR intervals (in seconds)
    rr_intervals = np.diff(rpeaks) / fs
    # Calculate Heart rate of Pateint
    heart_rate = 60 * len(rr_intervals) / np.sum(rr_intervals)
    return heart_rate



data_Insertor()    