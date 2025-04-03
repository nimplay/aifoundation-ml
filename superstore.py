import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Superstore.csv", encoding='latin1')

"""
df.head()
df.info()

# Filling null values
df['Postal Code'] = df['Postal Code'].fillna(0).astype(int)

# Deleting duplicated values
if df.duplicated().sum()> 0:
    print("There are duplicates in the dataset")
else:
    print("There are no duplicates in the dataset")
"""
# Types of customers
types_of_customers = df['Segment'].unique()
print(types_of_customers)

# Number of customers
number_of_customers = df['Segment'].value_counts().reset_index()
number_of_customers = number_of_customers.rename(columns={'Segment': 'Type Of Customers'})
print(number_of_customers)

#print pie distribution of customers

plt.pie(number_of_customers['count'], labels=number_of_customers['Type Of Customers'], autopct='%1.1f%%')
plt.show()
