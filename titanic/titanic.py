import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# TODO is staying?
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
sns.set()


"""
TODO:
- Update titanic.MD
- Add more options of different algorithms - when have better features
- Consider adding XGBoost

Feature ideas:
- Remove
"""


def read_files():
    x = pd.read_csv('input/train.csv', index_col='PassengerId')
    y = x.pop('Survived')
    x_test = pd.read_csv('input/test.csv', index_col='PassengerId')
#    print(f'x, shape: {x.shape}, head:\n{x.head()}')
#    print(f'y, shape: {y.shape}, head:\n{y.head()}')
#    print(f'x_train, shape: {x_test.shape}, head:\n{x_test.head()}')
    return x, y, x_test


def clean_handle_missing_categorical(x, columns_to_drop, age_for_missing, mean_class_3_fare):
    x.drop(columns_to_drop, axis=1, inplace=True)

    print(f'Features after dropping: {x.columns.values}')

    # Create a new feature of number of relatives regardless of who they are
    if 'SibSp' not in columns_to_drop and 'Parch' not in columns_to_drop:
        x['Family_Num'] = x['SibSp'] + x['Parch']
        x.drop(['SibSp', 'Parch'], axis=1, inplace=True)

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

    print(f'Number of members in family:\n{x["Family_Num"].value_counts().sort_index()}')


def scale_train(x_train):
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler


def output_preds(preds, x_test, name_suffix):
    pred_df = pd.DataFrame(preds, index=x_test.index, columns=['Survived'])
    pred_df.to_csv(f'output/preds_{name_suffix}.csv')


def cross_valid(classifier, x_train, y_train):
    accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
    return accuracies.mean(), classifier


def fit_single_classifier(classifier, x_train, y_train):
    classifier = classifier
    classifier.fit(x_train, y_train)
    return classifier.score(x_train, y_train), classifier


def grid_search(classifier, param_grid, x_train, y_train):
    grid = GridSearchCV(classifier, param_grid, verbose=1, cv=10, n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid.score(x_train, y_train), grid


def single_and_grid_classifier(name_str, x_train, y_train, x_test_local, y_test_local, single_classifier, grid_params):
    reg_score, classifier = cross_valid(single_classifier, x_train, y_train)
    print(f'{name_str} - Single train cross-validation classification score: {round(reg_score, 3)}')

    reg_score, classifier = grid_search(single_classifier, grid_params, x_train, y_train)
    print(f'{name_str} - Grid Search train classification score: {round(reg_score, 3)}, '
          f'local test score: {round(classifier.score(x_test_local, y_test_local), 3)}, '
          f'best classifier:\n{classifier.best_estimator_}')
    return classifier


def grid_with_voting(classifiers, param_grid, x_train, y_train, x_test_local, y_test_local):
    voting_classifier = VotingClassifier(estimators=classifiers, voting='soft', n_jobs=-1)

    grid = GridSearchCV(voting_classifier, param_grid, verbose=1, cv=10, n_jobs=-1)
    grid.fit(x_train, y_train)
    print(f'Grid Search With Voting classification train score: {round(grid.score(x_train, y_train), 3)}, '
          f'local test score: {round(grid.score(x_test_local, y_test_local), 3)}, '
          f'best classifier:\n{grid.best_estimator_}')
    return grid


def voting_only(classifiers, x_train, y_train, x_test_local, y_test_local, weights=None):
    voting_classifier = VotingClassifier(estimators=classifiers, voting='soft', n_jobs=-1, weights=weights)
    voting_classifier.fit(x_train, y_train)
    print(f'Best classifiers, Voting only classification train score: '
          f'{round(voting_classifier.score(x_train, y_train), 3)}, '
          f'local test score: {round(voting_classifier.score(x_test_local, y_test_local), 3)}, '
          f'best classifier:\n{voting_classifier}')
    return voting_classifier


def main():
    columns_to_drop = ['Name', 'Ticket', 'Cabin', 'Embarked']

    x, y, x_test = read_files()

    x_train, x_test_local, y_train, y_test_local = train_test_split(x, y, random_state=42)

    age_for_missing = x_train['Age'].mean()
    mean_class_3_fare = x_train[(x_train['Pclass'] == 1) | (x_train['Pclass'] == 2)]['Fare'].mean()

    clean_handle_missing_categorical(x_train, columns_to_drop, age_for_missing, mean_class_3_fare)
    clean_handle_missing_categorical(x_test_local, columns_to_drop, age_for_missing, mean_class_3_fare)
    clean_handle_missing_categorical(x_test, columns_to_drop, age_for_missing, mean_class_3_fare)

    scaler = scale_train(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_local_scaled = scaler.transform(x_test_local)
    x_test_scaled = scaler.transform(x_test)

    single_and_grid_classifier('Logistic - liblinear', x_train_scaled, y_train, x_test_local_scaled, y_test_local,
                               LogisticRegression(solver='liblinear', n_jobs=-1),
                               [{}])
    # Following was tried also for Logistic and didn't make it much better: solver=['lbfgs', 'newton-cg', 'sag', 'saga']

    single_and_grid_classifier('KNN - 14', x_train_scaled, y_train, x_test_local_scaled, y_test_local,
                               KNeighborsClassifier(n_jobs=-1, n_neighbors=14),
                               [{}])
    # Following was tried also and found 14 to be best: n_neighbors=[range(1, 25), 5 (lower scores), 25 (lower scored)]

    single_and_grid_classifier('SVM - rbf', x_train_scaled, y_train, x_test_local_scaled, y_test_local,
                               SVC(gamma='auto', kernel='rbf'),
                               [{
                                   'C': [0.5, 1.0, 1.5, 2.0],
                                   'gamma': [0.2, 0.1, 0.05, 0.01, 'auto_deprecated', 'scale']}
                                ])
    single_and_grid_classifier('SVM - poly', x_train_scaled, y_train, x_test_local_scaled, y_test_local,
                               SVC(gamma='auto', kernel='poly'),
                               [{
                                   'C': [0.5, 1.0, 1.5, 2.0],
                                   'gamma': [0.2, 0.1, 0.05, 0.01, 'auto_deprecated', 'scale']}
                                ])
    # sigmoid kerned was also tried for SVM, but gave worse results

    single_and_grid_classifier('NB', x_train_scaled, y_train, x_test_local_scaled, y_test_local,
                               GaussianNB(),
                               [{}])

    single_and_grid_classifier('RandomForest - 9', x_train_scaled, y_train, x_test_local_scaled, y_test_local,
                               RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=9),
                               [{}])
    '''
    for i in range(2, 21):
        reg_score, classifier = grid_search(RandomForestClassifier(max_depth=i, n_estimators=100),
                                            [{}],
                                            x_train, y_train)
        print(f'max_depth {i}: Grid Search train classification score: {round(reg_score, 3)}, '
              f'local test score: {round(classifier.score(x_test_local, y_test_local), 3)}, ')
    '''

    '''
    classifiers_all = [
        ('lr', LogisticRegression(solver='liblinear')),
        ('knn', KNeighborsClassifier()),
        ('svm', SVC(probability=True, gamma='auto')),
        ('nb', GaussianNB()),
        ('rf', RandomForestClassifier())
    ]

    grid_voting_params_all = [
        {'lr__solver': ['liblinear', 'lbfgs'],
         'knn__n_neighbors': [10, 14, 20],
         'svm__C': [0.1, 0.5, 1.0],
         'svm__gamma': [0.1, 0.05, 0.01, 'auto_deprecated', 'scale'],
         'svm__kernel': ['rbf', 'sigmoid', 'poly'],
         'svm__degree': [2, 3, 4],
         'rf__n_estimators': [100],
         'rf__max_depth': [9]}
    ]

    classifier_all = grid_with_voting(classifiers_all, grid_voting_params_all,
                                                 x_train_scaled, y_train, x_test_local_scaled, y_test_local)
    preds = classifier_all.predict(x_test_scaled)
    output_preds(preds, x_test, 'grid')
    '''

    classifiers_specific_with_params = [
        ('lr', LogisticRegression(solver='liblinear')),
        ('knn', KNeighborsClassifier(n_neighbors=14)),
        ('svm - rbf', SVC(probability=True, kernel='rbf', gamma=0.05, C=1.0)),
        ('svm - poly', SVC(probability=True, kernel='poly', gamma='auto_deprecated', C=0.5)),
        ('nb', GaussianNB()),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=9))
    ]

    classifier_voting = voting_only(classifiers_specific_with_params,
                                    x_train_scaled, y_train,
                                    x_test_local_scaled, y_test_local,
                                    [1, 1, 1, 1, 1, 0.3])

    preds = classifier_voting.predict(x_test_scaled)
    output_preds(preds, x_test, 'best')

main()
