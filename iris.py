import pandas as pd

df = pd.read_csv('iris.csv')
print(df.tail(3))

print(df['種類'].unique())
print(df['種類'].value_counts())
print(df.isnull().any(axis=0))
print(df.sum())
print(df.isnull().sum())
df2 = df.dropna(how='any', axis=0)
print(df2.tail(3))
