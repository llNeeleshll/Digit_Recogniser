import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from  keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

classifier = Sequential()

df = pd.read_csv('Data/train.csv')

df = df.fillna("0")

X = df.iloc[:, 1:].values

Y = df.iloc[:, 0:1].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

st_sc = StandardScaler()
x_train = st_sc.fit_transform(x_train)
x_test = st_sc.fit_transform(x_test)

# Using logistic Regression to start with

classifier.add(Dense(output_dim=700, init='uniform', activation='relu', input_dim=784))

classifier.add(Dense(output_dim=700, init='uniform', activation='relu'))

classifier.add(Dense(output_dim=700, init='uniform', activation='relu'))

classifier.add(Dense(output_dim=1, init='uniform', activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

modelfit = classifier.fit(x_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(x_test)

#kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

y_pred = (y_pred > 0.5)
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

pass

