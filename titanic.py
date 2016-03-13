import pandas as pd
# import numpy as np
from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import KFold
from sklearn import cross_validation
# from sklearn.ensemble import RandomForestClassifier

# opening train.csv file
data = pd.read_csv(open("/media/piyush/New Volume3/Titanic/train.csv", "rb"))
print data.head(5)
# Fill the na with median value
data["Age"] = data["Age"].fillna(data["Age"].median())
# print data.describe()									//gives overview of the data


# Convert the data to Numerical Values
data.loc[data["Sex"] == "male", "Sex"] = 0
data.loc[data["Sex"] == "female", "Sex"] = 1

data["Embarked"] = data["Embarked"].fillna("S")
data.loc[data["Embarked"] == "S", "Embarked"] = 0
data.loc[data["Embarked"] == "C", "Embarked"] = 1
data.loc[data["Embarked"] == "Q", "Embarked"] = 2


# Training
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# alg = LinearRegression()
# kf = KFold(data.shape[0], n_folds=3, random_state=1)

# predictions = []
# for train, test in kf:
#     train_predictions = (data[predictors].iloc[train, :])
#     train_target = data["Survived"].iloc[train]
#     alg.fit(train_predictions, train_target)
#     test_predictions = alg.predict(data[predictors].iloc[test, :])
#     predictions.append(test_predictions)

# predictions = np.concatenate(predictions, axis=0)

# predictions[predictions > 0.5] = 1
# predictions[predictions <= 0.5] = 0

# accuracy = sum(
# predictions[predictions == np.array(data["Survived"])]) /
# len(predictions)

# print sum(predictions[predictions == np.array(data["Survived"])]),  len(predictions)
# print accuracy

# Logistic Regression
alg = LogisticRegression(random_state=1)

sc = cross_validation.cross_val_score(
    alg, data[predictors], data["Survived"], cv=3)

print sc.mean()

# Processing test file

test = pd.read_csv(open("/media/piyush/New Volume2/Titanic/test.csv", "rb"))

test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1
test["Embarked"] = test["Embarked"].fillna("S")
test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

print test.describe()

alg.fit(data[predictors], data["Survived"])
predictions = alg.predict(test[predictors])

submissions = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})

# Writing the predictions to Kaggle.csv file
submissions.to_csv("kaggle.csv", index=False)
