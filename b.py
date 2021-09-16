import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print(model.predict([[30,60]]))
