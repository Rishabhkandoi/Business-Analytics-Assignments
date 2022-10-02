import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

def fitModel(data, n):
    train, test = train_test_split(data, test_size=0.2)
    logisticRegr = LogisticRegression()
    model = logisticRegr.fit(train.iloc[:, 0:n], train['Purchased'])
    pred = list(model.predict(test.iloc[:, 0:n]))
    fit = list(test['Purchased'])
    accuracy = accuracy_score(fit, pred)
    print("\n\nAccuracy: ")
    print(accuracy)
    print("\n\nConfusion Matrix: ")
    print(confusion_matrix(fit, pred))

data = pd.read_csv("C:/Users/hp/Documents/Google Drive NIIT/Semester 7/Business Analytics/Assignments/PythonAssign/Social_Network_Ads.csv")
data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})
Y = data['Purchased']
X = data.iloc[:, 0:4]
print("\n\nBefore doing feature selection, accuracy and confusion matrix of original dataset is as follows:\n")
fitModel(data, 4)
selector = RFE(LogisticRegression(), 3)
selector = selector.fit(X, Y)
print("\nFeature selection gives ranking of predictors as: ")
print(selector.ranking_)
print("\n\nTherefore, 'Gender' is found to be of least importance in our model. So after dropping that column, our result is as follows:")
data = data.drop(columns={"Gender"})
fitModel(data, 3)
print("\nTherefore most of the time, we got better result after removing one of the predictors.")
