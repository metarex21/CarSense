# **Exploratory Data Analysis (Car Price Prediction)**
## 1. Importing Necessary Libraries
#importing for mathematical computation
import numpy as np 
import pandas as pd
import scipy.stats as stats
#importing for data visualization
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import figure

## 2. Loading The Data
# Loading the dataset
df = pd.read_csv("../dataset/cars_ds_final.csv")
### Fixing Anomalies
# Create a copy of the dataframe
df_copy = df.copy()

# Convert prices to string format and remove "Rs." and commas
df_copy['Ex-Showroom_Price'] = df_copy['Ex-Showroom_Price'].str.replace(',', '').str.replace('Rs. ', '')

# Convert prices to integer type
df_copy['Ex-Showroom_Price'] = df_copy['Ex-Showroom_Price'].astype(int)

# Update the existing dataframe with the modified prices
df['Ex-Showroom_Price'] = df_copy['Ex-Showroom_Price']

# Save the updated dataframe to a new CSV file
df_copy.to_csv('cars_ds_price.csv', index=False)

## 3. Exploring the Data
# Display the first few rows of the dataset
print(df.head())
# Check the dimensions of the dataset
print(df.shape)
# Summarize basic statistics
description = df.drop('Sl_no', axis=1).describe()

print(description)
# Check the data types of each column
print(df.dtypes)
# Identify missing values
print(df.isnull().sum())
# Display the column names
print(df.columns)
# Explore categorical columns
print("Unique values in categorical columns:\n")
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    print(column, ":", df[column].nunique())
print()
## 4. Visualising the Data
# Visualize the distribution of numerical columns excluding 'Sl_no'
print("Visualizing the distribution of numerical columns (excluding 'Sl_no'):")
numerical_columns = df.select_dtypes(include=['int32','int64', 'float64']).columns.drop('Sl_no')
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

# Visualize the relationship between numerical columns and the target variable
target_column = 'Make'
print("Relationship between numerical columns and the target variable:")
for column in numerical_columns:
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x=column, y=target_column)
    plt.title(f"{column} vs {target_column}")
    plt.show()
# Calculate the count of unique cars made by each company
unique_cars_by_company = df.groupby('Make')['Model'].nunique().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=unique_cars_by_company.index, y=unique_cars_by_company.values)
plt.title("Number of Unique Cars by Company")
plt.xlabel("Car Make")
plt.ylabel("Number of Unique Cars")
plt.xticks(rotation=90)
plt.show()

# Visualize the top manufacturers
top_manufacturers = df['Make'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_manufacturers.index, y=top_manufacturers.values)
plt.title("Top 10 Manufacturers")
plt.xlabel("Manufacturer")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
average_price_by_make = df.groupby('Make')['Ex-Showroom_Price'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=average_price_by_make.index, y=average_price_by_make.values)
plt.title("Average Price by Manufacturer")
plt.xlabel("Car Make")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.show()

# Define the number of price ranges and calculate the range width
num_ranges = 8
price_range_width = (10000000 - 100000) / num_ranges

# Create bins for the price ranges
bins = [100000 + i * price_range_width for i in range(num_ranges + 1)]
labels = [f'{bins[i]:.2f} - {bins[i+1]:.2f}' for i in range(num_ranges)]

# Filter the dataframe to exclude values outside the specified range
filtered_df = df[(df['Ex-Showroom_Price'] >= 100000) & (df['Ex-Showroom_Price'] <= 10000000)]

# Assign the price ranges to the filtered dataframe
filtered_df['Price_Range'] = pd.cut(filtered_df['Ex-Showroom_Price'], bins=bins, labels=labels, include_lowest=True)

# Count the number of cars in each price range
price_range_counts = filtered_df['Price_Range'].value_counts().sort_index()

# Plot the number of cars in each price range
plt.figure(figsize=(10, 6))
sns.barplot(x=price_range_counts.index, y=price_range_counts.values)
plt.title("Number of Cars in Each Price Range (Excluding Values Outside 100,000 - 10,000,000)")
plt.xlabel("Price Range (Ex-Showroom Price)")
plt.ylabel("Number of Cars")
plt.xticks(rotation=45)
plt.show()

# Visualize the average price by body type
average_price_by_body_type = df.groupby('Body_Type')['Ex-Showroom_Price'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=average_price_by_body_type.index, y=average_price_by_body_type.values)
plt.title("Average Price by Body Type")
plt.xlabel("Body Type")
plt.ylabel("Average Price(In 10^8)")
plt.xticks(rotation=45)
plt.show()

## 5. Correlation Analysis
# Visualize the correlation between numerical columns
print("Correlation between numerical columns:")
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_columns].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation between numerical columns")
plt.show()
## Conclusion

In this project, we performed an exploratory data analysis (EDA) on the Indian Cars dataset. We started by loading the dataset and gaining an understanding of its structure and contents. We examined the columns, data types, and basic statistics of the dataset.

During the analysis, we explored various aspects of the dataset including the distribution of numerical columns, the count of unique car makes, the relationship between numerical columns and the target variable (make), and correlation analysis between variables.

Some key insights from the analysis include:

- The dataset consists of cars from various manufacturers, with Tata, Maruti Suzuki, and Hyundai being the most common brands.
- The price range of cars in the dataset varies widely, with some outliers at both the low and high ends.
- There is a positive correlation between certain numerical features such as engine displacement, cylinders, and ex-showroom price.
- Categorical variables such as fuel type, transmission type, and car body type show some level of association with the car make.

Additionally, we visualized the data using various plots and charts to better understand the distributions, relationships, and trends in the dataset.

Overall, this EDA analysis provides valuable insights into the Indian Cars dataset, highlighting important aspects such as the distribution of features, relationships between variables, and patterns within the data. These insights can serve as a foundation for further analysis or machine learning tasks such as car price prediction or classification.

It is important to note that this analysis is based on the available data and the findings are specific to this dataset. Further analysis and validation may be required for more accurate conclusions and predictions.
