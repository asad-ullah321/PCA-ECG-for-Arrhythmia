import psycopg2
import numpy as np
import os



# Connect to the database
conn = psycopg2.connect(
    dbname="arrhythmia_dataset_wavelet",
    user="postgres",
    password="axdw1234",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# File to store the IDs of deleted tuples
deleted_ids_file = "deleted_ids_wavelet.txt"
batch_size = 1000
offset = 0 
print("fecthing")
while True:
    print("fetching batch", offset/batch_size)
    # Fetch the IDs and n_signal from the ecg table
    cursor.execute(f"SELECT id, denoised_signal FROM ECG LIMIT {batch_size} OFFSET {offset}")
    rows = cursor.fetchall()
    print("fecthing done")

    if len(rows) == 0:
        print("breaking")
        break
    # List to store the IDs of tuples to be deleted
    ids_to_delete = []

    for row in rows:
        id, n_signal_bytes = row
        # Convert bytes to NumPy array
        n_signal = np.frombuffer(n_signal_bytes, dtype=float)
        
        # Check for NaN values
        if np.isnan(n_signal).any():
            ids_to_delete.append(id)

    # Delete the tuples with the marked IDs
    if ids_to_delete:
        # print("deleting")
        cursor.execute("DELETE FROM ecg WHERE id = ANY(%s)", (ids_to_delete,))
        conn.commit()

        # Save the IDs of the deleted tuples to a text file
        # Save the IDs of the deleted tuples to a text file in append mode
        with open(deleted_ids_file, "a") as file:   
            for id in ids_to_delete:
                print(id)
                file.write(f"{id}\n")
    offset+=batch_size

# Close the database connection
cursor.close()
conn.close()