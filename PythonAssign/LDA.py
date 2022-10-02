import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import FunctionTransformer

data = pd.read_csv("H:/Downloads/Wine.csv")
train, test = train_test_split(data, test_size=0.2)
logisticRegr = LinearDiscriminantAnalysis()
model = logisticRegr.fit(train.iloc[:, 0:13], train['Customer_Segment'])
train_tr = FunctionTransformer().fit_transform(train)
train_tr = pd.DataFrame(train_tr)
model_tr = logisticRegr.fit(train_tr.iloc[:, 0:13], train_tr.iloc[:,13])
pred = list(model.predict(test.iloc[:, 0:13]))
fit = list(test['Customer_Segment'])
accuracy = accuracy_score(fit, pred)
print("Before Transformation:\n")
print(accuracy)
print(confusion_matrix(fit, pred))

test_tr = FunctionTransformer().fit_transform(test)
test_tr = pd.DataFrame(test_tr)
pred_tr = list(model_tr.predict(test_tr.iloc[:, 0:13]))
fit_tr = list(test_tr.iloc[:,13])
accuracy_tr = accuracy_score(fit, pred_tr)
print("After Transformation:\n")
print(accuracy_tr)
print(confusion_matrix(fit_tr, pred_tr))

print("Therefore, Transformation in this dataset does not produce any changes.")
