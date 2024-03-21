from sklearn.decomposition import IncrementalPCA
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Initialize the PCA model
# ipca = IncrementalPCA(n_components=10, batch_size=100)

def create_conn():
    conn = psycopg2.connect(
         dbname="arrhythmia_dataset_2",
        user="postgres",
        password="axdw1234",
        host="localhost",
        port="5432"
    )
    return conn

def load_data_in_batches(batch_size=1000, offset=0):
    
    
    conn = create_conn()
    result = []
    # Load your data in batches
    cursor = conn.cursor()
    
    try:
        
        print("fetching batch", offset/batch_size)
        # Fetch the IDs and n_signal from the ecg table
        cursor.execute(f"SELECT id,n_signal FROM ECG LIMIT {batch_size} OFFSET {offset}")
        # cursor.execute(f"SELECT id,denoised_signal FROM ECG LIMIT {batch_size} OFFSET {offset}")

        rows = cursor.fetchall()
        print("fecthing done")
        if len(rows) == 0:
            print("breaking")
            return []
        # print(rows)
        for row in rows:
            data = np.frombuffer(row[1],dtype=float)
            data = data.reshape((5000,12))
            result.append(data)

        print(rows[0][0], rows[-1][0])
        return result

    except Exception as e:
        print(e)
    finally:
        cursor.close()
        conn.close()


    
   


def pca_():
    # Assuming you have your dataset loaded into a variable named 'ecg_data'
    # 'ecg_data' should be a numpy array with shape (num_patients, num_time_points_per_patient, num_features_per_time_point)

    # Reshape the data into a 2D array (num_samples, num_features)
    ipca = IncrementalPCA(n_components=4)  # Assuming you want to reduce to 2 dimensions

    # Define the batch size
    batch_size = 4000
    offset = 0
    while True:

        ecg_data = load_data_in_batches(batch_size=batch_size, offset=offset)
        ecg_data = np.array(ecg_data)
        if len(ecg_data) == 0:
            print("end")
            print("breaking")
            break
        num_patients, num_time_points_per_patient, num_features_per_time_point = ecg_data.shape
        print("Shape: ", num_patients, num_time_points_per_patient, num_features_per_time_point, "\n")
        ecg_data_reshaped = ecg_data.reshape(-1, num_features_per_time_point)

        # Standardize the data
        scaler = StandardScaler()
        ecg_data_scaled = scaler.fit_transform(ecg_data_reshaped)


        # Initialize Incremental PCA

        # Process data in batches
        for i in range(0, len(ecg_data_scaled), batch_size):
            batch = ecg_data_scaled[i:i+batch_size]
            ipca.partial_fit(batch)
        offset+=batch_size
    # Transform the entire dataset using the learned transformation
    # ecg_data_transformed = ipca.transform(ecg_data_scaled)
    # Get the principal components (eigenvectors)
    principal_components = ipca.components_
    
    # Get explained variance ratios
    explained_variance = ipca.explained_variance_ratio_
  # Plot explained variance for analysis (optional)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.grid(True)
    plt.show()

    # Find the number of components required to explain at least 90% of the variance
    cumulative_variance = np.cumsum(explained_variance)
    num_components = np.argmax(cumulative_variance >= 0.9) + 1

    print(f"Number of components required to explain at least 90% of the variance: {num_components}")
    ipca.n_components = num_components
    # tranform(ipca=ipca)



def tranform(ipca):
    
    batch_size = 1000
    offset =0
    conn = create_conn()
    result = []
    # Load your data in batches
    cursor = conn.cursor()
    print("Transformation")
    try:
       while True: 
            print("fetching batch", offset/batch_size)
            # Fetch the IDs and n_signal from the ecg table
            cursor.execute(f"SELECT id,n_signal FROM ECG LIMIT {batch_size} OFFSET {offset}")
            # cursor.execute(f"SELECT id,denoised_signal FROM ECG LIMIT {batch_size} OFFSET {offset}")

            rows = cursor.fetchall()
            print("fecthing done")
            if len(rows) == 0:
                print("breaking")
                return []
            # print(rows)
            for row in rows:
                data = np.frombuffer(row[1],dtype=float)
                data = data.reshape((5000,12))
                resultant_data = ipca.transform(data)
                print(resultant_data)
            offset+=batch_size

    


        

    except Exception as e:
        print(e)
    finally:
        cursor.close()
        conn.close()
    
pca_()


