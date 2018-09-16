import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

classifier = DecisionTreeClassifier()

df = pd.read_csv('Data/train.csv')

df = df.fillna("0")

X = df.iloc[:, 1:].values

Y = df.iloc[:, 0:1].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

modelfit = classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_pred, y_test)

corr = 0
incorr = 0

for i in range(0, len(y_test)):
    if y_test[i] == y_pred[i]:
        corr += 1
    else:
        incorr += 1

corrPer = (corr / len(y_test)) * 100

incorPer = (incorr / len(y_test)) * 100

# use the model for test file

df_test = pd.read_csv('Data/test.csv')

df_test = df_test.fillna("0")

x_maintest = df_test.iloc[:, 0:].values

y_mainpred = classifier.predict(x_maintest)


df_created = pd.DataFrame(y_mainpred, columns=['Class'])
df_created.to_csv("Data/pred.csv")

