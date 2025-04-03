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

# sales per segment
sales_per_segment = df.groupby('Segment')['Sales'].sum().reset_index()
sales_per_segment = sales_per_segment.rename(columns={'Segment': 'Type Of Customer', 'Sales': 'Total Sales'})

print(sales_per_segment)

# print sales per segment
plt.bar(sales_per_segment['Type Of Customer'], sales_per_segment['Total Sales'])
plt.show()
plt.pie(sales_per_segment['Total Sales'], labels=sales_per_segment['Type Of Customer'], autopct='%1.1f%%')
plt.show()

# customers order of frecuency
customer_order_frecuency = df.groupby(['Customer ID', 'Customer Name', 'Segment'])['Order ID'].count().reset_index()
customer_order_frecuency = customer_order_frecuency.rename(columns={'Order ID': 'Total Orders'})
repeat_customers = customer_order_frecuency[customer_order_frecuency['Total Orders'] >= 1]
repeat_customers_sorted = repeat_customers.sort_values(by='Total Orders', ascending=False)
print(repeat_customers_sorted.head(12).reset_index(drop=True))
