import pandas as pd

df = pd.read_csv('iris.csv')
print(df.tail(3))

print(df['種類'].unique())
print(df['種類'].value_counts())
print(df.isnull().any(axis=0))
print(df.sum())
print(df.isnull().sum())
# df['花弁長さ'] = df['花弁長さ'].fillna(0)
print('mean')
colmean = df.mean()
print(colmean)
df = df.fillna(colmean)
print(df.tail(3))
print(df.mean())
print(df.std())

print(df.isnull().any(axis=0))

