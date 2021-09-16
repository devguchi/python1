import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.csv')
df = df.fillna(df.mean())
x = df[['g_len','g_width','k_len','k_width']]
t = df['type']
x_train,x_test,y_train,y_test = train_test_split(
        x,t,test_size=0.3,random_state=0)

model = tree.DecisionTreeClassifier(
        max_depth=2, random_state=0)

model.fit(x_train,y_train)
print(model.score(x_test,y_test))

with open('iris.pkl', 'wb') as f:
    pickle.dump(model, f)

plt.figure(figsize=(15,10))
tree.plot_tree(
        model,feature_names=x_train.columns,filled=True)
plt.show()

