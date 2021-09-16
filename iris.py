import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.csv')
colmean = df.mean()
df = df.fillna(colmean)
xcol = ['がく片長さ','がく片幅','花弁長さ','花弁幅']
x = df[xcol]
t = df['種類']
x_train,x_test,y_train,y_test = train_test_split(
        x,t,test_size=0.3,random_state=0)

model = tree.DecisionTreeClassifier(
        max_depth=2, random_state=0)

model.fit(x_train,y_train)
print(model.score(x_test,y_test))

with open('iris.pkl', 'wb') as f:
    pickle.dump(model, f)

print(model.tree_.feature)
print(model.tree_.threshold)
print(model.tree_.value[1])
print(model.tree_.value[3])
print(model.tree_.value[4])
print(model.classes_)

x_train.columns = ['glen','gwith','klen','kwidth']
plt.figure(figsize=(15,10))
tree.plot_tree(
        model,feature_names=x_train.columns,filled=True)
plt.show()
