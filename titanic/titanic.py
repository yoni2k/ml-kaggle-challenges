import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
sns.set()


def read_files():
    x_train = pd.read_csv('input/train.csv', index_col='PassengerId')
    y_train = x_train.pop('Survived')
    x_test = pd.read_csv('input/test.csv', index_col='PassengerId')
#    print(f'x_train, shape: {x_train.shape}, head:\n{x_train.head()}')
#    print(f'y_train, shape: {y_train.shape}, head:\n{y_train.head()}')
#    print(f'x_train, shape: {x_test.shape}, head:\n{x_test.head()}')
    return x_train, y_train, x_test


def clean_handle_missing_categorical(x, columns_to_drop, age_for_missing, mean_class_3_fare):
    x.drop(columns_to_drop, axis=1, inplace=True)

    # Split 3 categorical unique values (1, 2, 3) of Pclass into 2 dummy variables for classes 1 & 2
    if 'Pclass' not in columns_to_drop:
        x['pclass_1'] = x['Pclass'].apply(lambda cl: 1 if cl == 1 else 0)
        x['pclass_2'] = x['Pclass'].apply(lambda cl: 1 if cl == 2 else 0)
        x.drop('Pclass', axis=1, inplace=True)

    # Change categorical feature 'Sex' to be 1 encoded 'Male', 1 = Male, 0 = Fembale
    if 'Sex' not in columns_to_drop:
        x['Male'] = x['Sex'].map({'male': 1, 'female': 0})
        x.drop('Sex', axis=1, inplace=True)

    ''' 
        Change categorical feature 'Embarked' with 3 values ('S', 'C', 'Q') to be 2 dummy variables: 
            Embarked_S, Embarked_Q with 'C' being a reference variable
            In addition, handle 2 missing values to have them the most common value 'S'
    '''
    if 'Embarked' not in columns_to_drop:
        x['Embarked_S'] = x['Embarked'].map({np.NaN: 1, 'S': 1, 'C': 0, 'Q': 0})
        x['Embarked_Q'] = x['Embarked'].map({np.NaN: 0, 'S': 0, 'C': 0, 'Q': 1})
        x.drop('Embarked', axis=1, inplace=True)

    if 'Age' not in columns_to_drop:
        x['Age'].replace({np.NaN: age_for_missing}, inplace=True)

    if 'Fare' not in columns_to_drop:
        x['Fare'].replace({np.NaN: mean_class_3_fare}, inplace=True)


def scale_train(x_train):
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler


# TODO is staying?
def single_logistic_regression(x_train, y_train):
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    return log_reg.score(x_train, y_train), log_reg


def output_preds(preds, x_test):
    pred_df = pd.DataFrame(preds, index=x_test.index)
    pred_df.to_csv('output/preds.csv')


# TODO is staying?
def grid_search(classifier, param_grid, x_train, y_train):
    grid = GridSearchCV(classifier, param_grid, verbose=1, cv=10, n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid.score(x_train, y_train), grid


def grid_with_voting(classifiers, param_grid, x_train, y_train):
    voting_classifier = VotingClassifier(estimators=classifiers, voting='soft')

    grid = GridSearchCV(voting_classifier, param_grid, verbose=1, cv=10, n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid.score(x_train, y_train), grid


columns_to_drop = ['Name', 'Ticket', 'Cabin']
classifiers = [('lr', LogisticRegression())]
grid_voting_params = [{'lr__solver': ['liblinear', 'newton-cg', 'sag', 'saga']},
                      {'lr__solver': ['lbfgs']}]
grid_params = {'solver': ['liblinear', 'newton-cg', 'sag', 'saga']}


x_train, y_train, x_test = read_files()

age_for_missing = x_train['Age'].mean()
mean_class_3_fare = x_train[(x_train['Pclass'] == 1) | (x_train['Pclass'] == 2)]['Fare'].mean()

clean_handle_missing_categorical(x_train, columns_to_drop, age_for_missing, mean_class_3_fare)
clean_handle_missing_categorical(x_test, columns_to_drop, age_for_missing, mean_class_3_fare)

scaler = scale_train(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

'''
# TODO is staying?
reg_score, classifier = single_logistic_regression(x_train_scaled, y_train)
print(f'Single Linear Regression score: {reg_score}')
# TODO is staying?
reg_score, classifier = grid_search(LogisticRegression(), grid_params, x_train_scaled, y_train)
print(f'Grid Search Regression score: {reg_score}')
'''
reg_score, classifier = grid_with_voting(classifiers, grid_voting_params, x_train_scaled, y_train)
print(f'Grid Search With Voting Regression score: {reg_score}')

preds = classifier.predict(x_test_scaled)

output_preds(preds, x_test)


'''
Old code currently not used:
def cross_valid(classifier, x_train, y_train):
    accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
    print(f'Cross validation, mean: {accuracies.mean()}, STD: {accuracies.std()}')
cross_valid(classifier, x_train_scaled, y_train)
'''



