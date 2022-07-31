import model_evaluation
import pickle
import pandas as pd

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


sep_len = input('Enter Sepal Length : ')
sep_wid = input('Enter Sepal Width : ')
pet_len = input('Enter Petal Length : ')
pet_wid = input('Enter Petal Width : ')

pred_df = pd.DataFrame([[sep_len, sep_wid, pet_len, pet_wid]], columns = model_evaluation.x_test.columns)
pred = model.predict(pred_df)

targets = {0:'Setosa', 1:'Versicolor', 2:'Virginica'}

print(f'{"-"*120}\nPrediction is >>  Iris -', targets[pred[0]])