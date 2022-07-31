from sklearn.linear_model import LogisticRegression
from EDA import *
from sklearn.model_selection import train_test_split



df = load_data(load_iris())

x = df.drop('Target', axis=1)
y = df['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=10, test_size=0.2)

log_model = LogisticRegression()
log_model.fit(x_train, y_train)


if __name__ == '__main__':
    print(log_model.fit(x_train, y_train))