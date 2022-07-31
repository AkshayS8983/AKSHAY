import model_training
from model_training import log_model, x_test, y_test
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

y_pred_test = log_model.predict(x_test)


with open('model.pkl', 'wb') as f:
    pickle.dump(log_model, f)


if __name__=='__main__':
    print('Accuracy Score : ', accuracy_score(y_test, y_pred_test))
    print('-'*100)

    print('Classification Report :\n', classification_report(y_test, y_pred_test))
    print('-'*100)

    print('Confusion Matrix :\n', confusion_matrix(y_test, y_pred_test))