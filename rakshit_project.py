import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from a CSV file into a DataFrame
dataframe = pd.read_csv("bright.csv")

# 1. Display the top five rows of the DataFrame
print("\t\t1. TOP FIVE ROWS ARE:\n\n", dataframe.head())
# Conclusion: Displaying the first five rows provides an initial look at the data, including the structure and some sample values.

print("\n\n**************************************************************************\n\n")

# 2. Display the last five rows of the DataFrame
print("\t\t\t2. LAST FIVE ROWS ARE:\n\n", dataframe.tail())
# Conclusion: Displaying the last five rows helps understand the dataset's end values and identify if there is any trailing data irregularity.

print("\n\n**************************************************************************\n\n")

# 3. Display the shape of the DataFrame
print("3. SHAPE OF DATASET IS:\t", dataframe.shape, end=" ")
# Conclusion: The shape method reveals the number of rows and columns, which is essential for understanding the dataset's size and dimensionality.

print("\n\n**************************************************************************\n\n")

# 4. Display the data types of each feature in the DataFrame
print("4. DATATYPES OF EACH FEATURE ARE:\n\n", dataframe.dtypes)
# Conclusion: Knowing the data types helps in understanding the kind of operations that can be performed on each column and if any type conversion is necessary.

print("\n\n**************************************************************************\n\n")

# 5. Display the statistical summary of the DataFrame, including all columns
print("\t\t\t5. STATISTICAL SUMMARY IS:\n\n", dataframe.describe(include='all'))
# Conclusion: The statistical summary provides insights into the central tendency, dispersion, and shape of the dataset's distribution, including count, mean, std, min, and max values.

print("\n\n**************************************************************************\n\n")

# 6. Check for null values in the DataFrame
print("6. TO CHECK FOR NULL VALUES:\n\n", dataframe.isnull().sum())
# Conclusion: Summarizing null values helps in identifying missing data, which is crucial for data cleaning and preprocessing steps.

print("\n\n**************************************************************************\n\n")

# 7. Check for duplicate values in the DataFrame
print("7. TO CHECK FOR DUPLICATE VALUES:\n\n", dataframe.duplicated())
# Conclusion: Identifying duplicate rows ensures data uniqueness and integrity, avoiding any potential bias in analysis or modeling.

print("\n\n**************************************************************************\n\n")

# 8. Check for anomalies in specific columns of the DataFrame
print("8. TO CHECK FOR ANOMALIES VALUES:\n")
# Checking for anomalies in the 'Age' column (values should be positive)
if (dataframe['Age'] <= 0).any():
    print("Anomaly detected in 'Age' column")

# Checking for anomalies in the 'Gender' column (values should be 'male' or 'female')
if not dataframe['Gender'].isin(['male', 'female']).all():
    print("Anomaly detected in 'Gender' column")

# Checking for anomalies in the 'Marital_status' column (values should be 'married' or 'single')
if not dataframe['Marital_status'].isin(['married', 'single']).all():
    print("Anomaly detected in 'Marital_status' column")

# Checking for anomalies in the 'Education' column (values should be 'Graduate' or 'Post Graduate')
if not dataframe['Education'].isin(['Graduate', 'Post Graduate']).all():
    print("Anomaly detected in 'Education' column")

# Checking for anomalies in the 'No_of_Dependents' column (values should be non-negative integers)
if dataframe['No_of_Dependents'].dtype != 'int64' or (dataframe['No_of_Dependents'] < 0).any():
    print("Anomaly detected in 'No_of_Dependents' column")

# Checking for anomalies in the 'Personal_loan' column (values should be 'Yes' or 'No')
if not dataframe['Personal_loan'].isin(['Yes', 'No']).all():
    print("Anomaly detected in 'Personal_loan' column")

# Checking for anomalies in the 'House_loan' column (values should be 'Yes' or 'No')
if not dataframe['House_loan'].isin(['Yes', 'No']).all():
    print("Anomaly detected in 'House_loan' column")

# Checking for anomalies in the 'Partner_working' column (values should be 'Yes' or 'No')
if not dataframe['Partner_working'].isin(['Yes', 'No']).all():
    print("Anomaly detected in 'Partner_working' column")

# Checking for anomalies in the 'Salary' column (values should be non-negative integers)
if dataframe['Salary'].dtype != 'int64' or (dataframe['Salary'] < 0).any():
    print("Anomaly detected in 'Salary' column")

# Checking for anomalies in the 'Partner_salary' column (values should be non-negative, allowing for NaN)
if (dataframe['Partner_salary'].dropna() < 0).any():
    print("Anomaly detected in 'Partner_salary' column")

# Checking for anomalies in the 'Total_salary' column (should be the sum of 'Salary' and 'Partner_salary')
if not dataframe[dataframe['Partner_salary'].notna()]['Total_salary'].equals(dataframe[dataframe['Partner_salary'].notna()]['Salary'] + dataframe[dataframe['Partner_salary'].notna()]['Partner_salary']):
    print("Anomaly detected in 'Total_salary' column")

# Checking for anomalies in the 'Price' column (values should be non-negative)
if (dataframe['Price'] < 0).any():
    print("Anomaly detected in 'Price' column")

# Checking for anomalies in the 'Make' column (values should not be null or empty)
if dataframe['Make'].isnull().any() or (dataframe['Make'] == '').any():
    print("Anomaly detected in 'Make' column")

# Conclusion: Identifying anomalies ensures that the data conforms to expected patterns and values, which is critical for accurate analysis and modeling.

print("\n\n**************************************************************************\n\n")

# 9. Check for outliers in numerical columns
print("9. TO CHECK FOR OUTLIERS:\n")

def detect_outliers(df, column):
    # Calculate the Interquartile Range (IQR)
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    # Define lower and upper bounds for detecting outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Identify outliers as values outside the bounds
    outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
    return outliers

numerical_columns = ['Age', 'Salary', 'Partner_salary', 'Total_salary', 'Price']

for column in numerical_columns:
    outliers = detect_outliers(dataframe, column)
    print(f"Outliers in {column}:")
    print(outliers)

# Conclusion: Detecting outliers helps in identifying extreme values that may skew the data analysis and results, ensuring more robust and accurate insights.

print("\n\n**************************************************************************\n\n")

# 10. Data Cleaning Steps
print("10. CLEANING STEPS:\n\n")

# A. Drop duplicate rows
print("\nA. DROP DUPLICATE ROWS\n\n")
dataframe_cleaned = dataframe.drop_duplicates()
print(f"Number of rows after dropping duplicates: {dataframe_cleaned.shape[0]}")
# Conclusion: Removing duplicate rows ensures the dataset's uniqueness and integrity, preventing redundant data from affecting analysis.

# B. Check for missing values in the cleaned DataFrame
print("\nB. CHECK FOR MISSING VALUES\n\n")
print(dataframe_cleaned.isnull().sum())
# Conclusion: Checking for missing values in the cleaned dataset ensures that no data is lost and identifies areas that may need further attention.

# C. Treating outliers in the numerical columns
print("\nC. TREATING OUTLIERS\n\n")
for column in numerical_columns:
    Q1 = dataframe_cleaned[column].quantile(0.25)
    Q3 = dataframe_cleaned[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Cap outliers at the lower and upper bounds
    dataframe_cleaned[column] = np.where(dataframe_cleaned[column] > upper_bound, upper_bound, 
                                         np.where(dataframe_cleaned[column] < lower_bound, lower_bound, dataframe_cleaned[column]))
print("SUCCESSFULLY TREATED OUTLIERS")
# Conclusion: Treating outliers by capping them at the lower and upper bounds reduces their impact on the analysis, leading to more stable and reliable results.

# D. Display summary statistics of the cleaned DataFrame
print("\n\t\t\tD. SUMMARY STATS\n\n")
print(dataframe_cleaned.describe())
# Conclusion: The summary statistics of the cleaned dataset provide a final overview of the data, ensuring that it is now ready for further analysis and modeling.

print("\n\n**************************************************************************")
print("**************************************************************************")
print("**************************************************************************")
print("**************************************************************************\n\n")

# Descriptive Statistics
# Calculate mean, median, and standard deviation for the 'Age' column
mean_age = dataframe['Age'].mean()
median_age = dataframe['Age'].median()
std_age = dataframe['Age'].std()
print(f"Mean Age: {mean_age}\nMedian Age: {median_age}\nStandard Deviation of Age: {std_age}")
print("\n\n**************************************************************************\n\n")

# The mean, median, and standard deviation provide a summary of the age distribution.
# Mean gives the average age, median gives the middle value, and standard deviation indicates variability.

# Correlation Analysis
# Calculate the correlation coefficient between 'Age' and 'Salary'
correlation_age_salary = dataframe[['Age', 'Salary']].corr().iloc[0, 1]
print(f"Correlation between Age and Salary: {correlation_age_salary}")
print("\n\n**************************************************************************\n\n")

# The correlation coefficient quantifies the linear relationship between age and salary.
# A value close to 1 or -1 indicates a strong relationship, while a value close to 0 indicates a weak relationship.

# Salary Analysis
# Calculate the average salary for each education level
average_salary_by_education = dataframe.groupby('Education')['Salary'].mean()
print("Average Salary by Education:\n\n", average_salary_by_education)
print("\n\n**************************************************************************\n\n")

# This shows how educational qualifications impact average salary, providing insights into salary expectations based on education.

# Loan Status
# Calculate the percentage of individuals with a personal loan
personal_loan_percentage = dataframe['Personal_loan'].value_counts(normalize=True) * 100
print("Percentage of Individuals with Personal Loan:\n\n", personal_loan_percentage)
print("\n\n**************************************************************************\n\n")

# Compare personal loan percentages between genders
personal_loan_by_gender = dataframe.groupby('Gender')['Personal_loan'].value_counts(normalize=True).unstack() * 100
print("Personal Loan by Gender:\n\n", personal_loan_by_gender)
print("\n\n**************************************************************************\n\n")

# These metrics show how common personal loans are and whether there are gender differences in taking personal loans.

# Marital Status and Dependents
# Convert '?' to NaN in 'No_of_Dependents' column
dataframe['No_of_Dependents'] = pd.to_numeric(dataframe['No_of_Dependents'], errors='coerce')

# Calculate the average number of dependents for each marital status, ignoring NaN values
average_dependents_by_marital_status = dataframe.groupby('Marital_status')['No_of_Dependents'].mean()

# Print the results
print("Average Number of Dependents by Marital Status:\n\n", average_dependents_by_marital_status)
print("\n\n**************************************************************************\n\n")

# Partner Employment and Total Salary
average_total_salary_by_partner_working = dataframe.groupby('Partner_working')['Total_salary'].mean()
print("Average Total Salary by Partner Working Status:\n\n", average_total_salary_by_partner_working)
print("\n\n**************************************************************************\n\n")

# Salary Comparison
average_salary_by_partner_working = dataframe.groupby('Partner_working')['Salary'].mean()
print("Average Salary by Partner Working Status:\n\n", average_salary_by_partner_working)
print("\n\n**************************************************************************\n\n")

# House Loan Analysis
house_loan_by_profession = dataframe.groupby('Profession')['House_loan'].value_counts(normalize=True).unstack() * 100
print("House Loan by Profession:\n\n", house_loan_by_profession)
print("\n\n**************************************************************************\n\n")

# Salary Distribution with Personal Loans
plt.figure(figsize=(10, 6))
sns.boxplot(x='Personal_loan', y='Salary', data=dataframe)
plt.title("Salary Distribution with Personal Loans")
plt.show()

# Automobile Make Analysis
# Replace ' ' (blank) in 'Salary' with NaN and convert to numeric
dataframe['Salary'] = pd.to_numeric(dataframe['Salary'].replace({' ': pd.NA}), errors='coerce')

# Calculate the average salary by automobile make
average_salary_by_make = dataframe.groupby('Make')['Salary'].mean()

# Print the results
print("Average Salary by Automobile Make:\n\n", average_salary_by_make)
print("\n\n**************************************************************************\n\n")


# Price Analysis
average_price = dataframe['Price'].mean()
print(f"Average Price of the Product/Service: {average_price}")
print("\n\n**************************************************************************\n\n")

price_by_total_salary = dataframe.groupby('Total_salary')['Price'].mean()
print("Price by Total Salary:\n\n", price_by_total_salary)
print("\n\n**************************************************************************\n\n")

# Marital Status and Loans
personal_loans_by_marital_status = dataframe.groupby('Marital_status')['Personal_loan'].value_counts(normalize=True).unstack() * 100
print("Personal Loans by Marital Status:\n\n", personal_loans_by_marital_status)
print("\n\n**************************************************************************\n\n")

# Educational Qualification Impact
house_loan_by_education = dataframe.groupby('Education')['House_loan'].value_counts(normalize=True).unstack() * 100
print("House Loan by Education:\n\n", house_loan_by_education)
print("\n\n**************************************************************************\n\n")

# Dependent Count Analysis
average_dependents_by_profession = dataframe.groupby('Profession')['No_of_Dependents'].mean()
print("Average Number of Dependents by Profession:\n\n", average_dependents_by_profession)
print("\n\n**************************************************************************\n\n")

# Gender and Salary
salary_by_gender = dataframe.groupby('Gender')['Salary'].mean()
print("Average Salary by Gender:\n\n", salary_by_gender)
print("\n\n**************************************************************************\n\n")

# Select relevant columns
data = dataframe[['Age', 'Education', 'No_of_Dependents', 'Salary']]
# Handle missing values (if any)
data = data.dropna()
# Encode categorical variable 'Education' if needed
data = pd.get_dummies(data, columns=['Education'], drop_first=True)

# Split data into features (X) and target (y)
X = data[['Age', 'Education_Post Graduate', 'No_of_Dependents']]
y = data['Salary']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize linear regression model
regressor = LinearRegression()
# Train the model using the training sets
regressor.fit(X_train, y_train)
# Predicting on the test set
y_pred = regressor.predict(X_test)
# Model evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error (MSE): {mse}\n")
print(f"R-squared (R2) Score: {r2}\n")

# Print the coefficients and intercept
print("\nModel Coefficients:\n")
for feature, coef in zip(X.columns, regressor.coef_):
    print(f"{feature}: {coef}\n")
print(f"Intercept: {regressor.intercept_}\n")

# Print the results
print(f"Manual Linear Regression MSE: {mse}, R2 Score: {r2}")
print("\n\n**************************************************************************\n\n")

# Loan Status Impact
total_salary_by_personal_loan = dataframe.groupby('Personal_loan')['Total_salary'].mean()
print("Total Combined Salary by Personal Loan Status:\n\n", total_salary_by_personal_loan)
print("\n\n**************************************************************************\n\n")

# Partner's Salary Contribution
average_partner_salary_by_house_loan = dataframe.groupby('House_loan')['Partner_salary'].mean()
print("Average Partner's Salary by House Loan Status:\n\n", average_partner_salary_by_house_loan)
print("\n\n**************************************************************************\n\n")

# Total Salary Distribution
plt.figure(figsize=(10, 6))
plt.hist(dataframe['Total_salary'], bins=20, edgecolor='k')
plt.title("Total Combined Salary Distribution")
plt.xlabel("Total Salary")
plt.ylabel("Frequency")
plt.show()