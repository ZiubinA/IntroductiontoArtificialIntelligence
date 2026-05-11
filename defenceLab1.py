import pandas as pd 

df = pd.read_csv("def.csv")
print(df['Gender'].nunique())
columns = ['Gender','City', 'Month Salary','Debt','Code']

for col in columns:
    print(f"percentef of missing values in the {col}",round(df[col].isnull().mean() * 100, 2))

for col in columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled missing values in {col} with median-{median_val}")

for col in columns:
    print(f"cardinality of columnn{col}", df[col].nunique())
print(df)
dfDrop = df['Gender'].drop_duplicates()
print(df['Gender'].drop_duplicates())
one = df[df['Gender'] == '1']
print(one['Month Salary'].mean())
zero = df[df['Gender'] == '0']
print(one['Month Salary'].mean())

df['Gender'] = df["Gender"].str.replace("l", "1").str.replace("M", "1").str.replace("F", "0").str.replace("o", "0")
print(df["Gender"].nunique())
outlier_columns = ['Month Salary', 'Debt', 'Code']
for col in outlier_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    #Calculates safe limits
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    original_count = len(df)
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    removed_count = original_count - len(df)
    
    if removed_count > 0:
        print(f"Removed {removed_count} outliers from {col}")

Vil = df[df['City'] == 'Vilnius']
Kau = df[df['City'] == 'Kaunas']
Kla = df[df['City'] == 'Klaipeda']

print("mean salary in vilnius is ", Vil['Month Salary'].mean())
print("mean salary in Kaunas is ", Kau['Month Salary'].mean())
print("mean salary in Klaipeda is ", Kla['Month Salary'].mean())

#task 5
df['City'] = df["City"].str.replace("Vilnius", "1").str.replace("Klaipeda", "3").str.replace("Kaunas", "2")
print(df['City'])

correlation_value = df['Month Salary'].corr(df['Debt'])
correlation_value2 = df['Month Salary'].corr(df['Code'])
correlation_value3 = df['Debt'].corr(df['Code'])
print("corelation value betwean salary debt, salary code, code dept", correlation_value, correlation_value2, correlation_value3)