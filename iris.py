import pandas as pd
from sklearn import tree

df = pd.read_csv('iris.csv')
colmean = df.mean()
df = df.fillna(colmean)
xcol = ['がく片長さ','がく片幅','花弁長さ','花弁幅']
x = df[xcol]
t = df['種類']

model = tree.DecisionTreeClassifier(
        max_depth=2, random_state=0)

model.fit(x,t)
print(model.score(x,t))
