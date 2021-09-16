import pandas as pd
from sklearn import tree

df = pd.read_csv('data.csv')
x = df[['age','score']]
t = df['money']

model = tree.DecisionTreeClassifier(random_state = 0)
model.fit(x, t)

a = [34,55]
b = [12,60]
c = [25,30]
d = [30,70]
print(model.predict([a,b,c,d]))
print(model.score(x,t))
