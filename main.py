import pandas as pd

df = pd.read_csv("heart.csv")


ds = df.copy()
ds = ds[ds["trestbps"] > 160]
print(ds)

def calculate_mean(df, name):
    return df[name].agg('mean')

def minMax(df, name):
    return df[name].agg(
        {
            "min" : 'min',
            "max" : 'max'
        })

def countValue(df, name):
    return df[name].agg('count')

def calCardinality(df, name):
    return df[name].nunique()

def calMedian(df, name):
    return df[name].median()

def calStd(df, name):
    return df[name].std()

def calPercentOfMissingVal(df, name):
    return df[name].isnull().mean() * 100
averageAge = calculate_mean(df, "age")
averagetrestbps = calculate_mean(df, "trestbps")
print("Average Age:", averageAge)
print("Average Age:", averagetrestbps)

maxMinAge = minMax(df, "age")
print(maxMinAge)

countRecords = countValue(df, "age")
print(countRecords)
print("cardinality of age", calCardinality(df, "age"))
print("median of age", calMedian(df, "age"))
print("standard deviation of age", calStd(df, "age"))
print("percenteg of missing values of age", calPercentOfMissingVal(df, "age"))
print(df)