
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_conn():
    conn = psycopg2.connect(
        database="arrhythmia_dataset_2",
        user="postgres",
        password="axdw1234",
        host="localhost",
        port="5432",
    )
    return conn


# Load the CSV file containing disease names and labels
# Assuming the CSV has columns 'label' and 'name'
# Convert the disease names DataFrame to a dictionary for easy lookup
disease_names_df = pd.read_csv('./dataset/dataset/ConditionNames_SNOMED-CT.csv')
disease_names_dict = disease_names_df.set_index('Snomed_CT')['Full Name'].to_dict()


conn = create_conn()
# Execute the query to fetch the data
query = """
SELECT id, age, gender, dx, hr
FROM ecg 
"""
data_df = pd.read_sql_query(query, conn)

# Close the database connection
conn.close()

# Split the disease labels, convert them to integers, and map to names
data_df['disease_labels'] = data_df['dx'].str.split(',').apply(lambda x: [int(label) for label in x])
data_df['disease_names'] = data_df['disease_labels'].apply(lambda x: [disease_names_dict.get(label, 'Unknown') for label in x])

print(type(data_df['disease_labels'][0]), "\n",data_df['disease_names'])


# Provide the labels of the diseases you're interested in
# selected_disease_labels = [
#     270492004,  # 1AVB
#     233917008,  # 2AVB
#     54016002,   # 2AVB1
#     28189009,   # 2AVB2
#     27885002,   # 3AVB
#     251173003,  # ABI
#     284470004,  # APB
#     39732003,   # ALS
#     426995002,  # JEB
#     251164006,  # JPT
#     11157007,   # VB
#     75532003,   # VEB
#     195060002,  # VPE
#     251180001   # VET
# ]

# # Filter the data to include only the selected diseases
# filtered_data_df = data_df[data_df['disease_labels'].apply(lambda x: any(label in x for label in selected_disease_labels))]
filtered_data_df = data_df

print(filtered_data_df.shape)

# Explode the list of disease names into separate rows
exploded_data_df = filtered_data_df.explode('disease_names').reset_index(drop=True)

# Plot the violin plot with diseases on the x-axis and heart rate on the y-axis
plt.figure(figsize=(12, 8))
sns.violinplot(x='disease_names', y='hr', data=exploded_data_df)
plt.title('Distribution of Heart Rate across Different Disease Labels')
plt.xlabel('Disease Label')
plt.ylabel('Heart Rate')
plt.xticks(rotation=90) # Rotate x-axis labels for better readability
plt.show()



#################################################################################
####################################AGE##########################################
#################################################################################



# Bin the ages into groups
age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # Adjust the bins as needed
age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
exploded_data_df['age_group'] = pd.cut(exploded_data_df['age'], bins=age_bins, labels=age_labels)

# Plot the violin plot with age groups on the y-axis and disease names on the x-axis
plt.figure(figsize=(12, 8))
sns.violinplot(x='disease_names', y='age_group', data=exploded_data_df)
plt.title('Distribution of Age Groups across Different Disease Labels')
plt.xlabel('Disease Label')
plt.ylabel('Age Group')
plt.xticks(rotation=90) # Rotate x-axis labels for better readability
plt.show()

#################################################################################
####################################avg hr by age################################
#################################################################################

# Group the data by 'Age' and calculate the average heart rate for each age group
age_groups = data_df.groupby('age')['hr'].mean()

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(age_groups.index, age_groups.values, marker='o', linestyle='-')
plt.title('Average Heart Rate by Age Group')
plt.xlabel('Age')
plt.ylabel('Average Heart Rate')
plt.grid(True)
plt.show()
