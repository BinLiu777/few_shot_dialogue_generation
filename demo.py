import pickle

with open('dialogs_test.pkl', 'rb') as f:
    fr = pickle.load(f)
    print(fr)

