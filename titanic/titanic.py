import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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
- Add more options of different algorithms
- Think about how Voting and GridSearch work together, consider removing some values
- Consider reading RandomForest - seems to be doing overfitting when added - need to split test set to really learn
"""


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


def output_preds(preds, x_test):
    pred_df = pd.DataFrame(preds, index=x_test.index, columns=['Survived'])
    pred_df.to_csv('output/preds.csv')


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


def single_and_grid_classifier(name_str, x_train, y_train, single_classifier, grid_params):
    reg_score, classifier = cross_valid(single_classifier, x_train, y_train)
    print(f'{name_str} - Single classification score: {reg_score}, default classifier:\n{classifier}')

    reg_score, classifier = grid_search(single_classifier, grid_params, x_train, y_train)
    print(f'{name_str} - Grid Search classification score: {reg_score}, best classifier:\n{classifier.best_estimator_}')


def grid_with_voting(classifiers, param_grid, x_train, y_train):
    voting_classifier = VotingClassifier(estimators=classifiers, voting='soft', n_jobs=-1)

    grid = GridSearchCV(voting_classifier, param_grid, verbose=1, cv=10, n_jobs=-1)
    grid.fit(x_train, y_train)
    print(f'Best estimator: {grid.best_estimator_}')
    return grid.score(x_train, y_train), grid


def voting_only(classifiers, x_train, y_train):
    voting_classifier = VotingClassifier(estimators=classifiers, voting='soft', n_jobs=-1)
    voting_classifier.fit(x_train, y_train)
    return voting_classifier.score(x_train, y_train), voting_classifier


def main():
    columns_to_drop = ['Name', 'Ticket', 'Cabin']

    x_train, y_train, x_test = read_files()

    age_for_missing = x_train['Age'].mean()
    mean_class_3_fare = x_train[(x_train['Pclass'] == 1) | (x_train['Pclass'] == 2)]['Fare'].mean()

    clean_handle_missing_categorical(x_train, columns_to_drop, age_for_missing, mean_class_3_fare)
    clean_handle_missing_categorical(x_test, columns_to_drop, age_for_missing, mean_class_3_fare)

    scaler = scale_train(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    single_and_grid_classifier('Logistic', x_train_scaled, y_train,
                               LogisticRegression(solver='liblinear', n_jobs=-1),
                               [{'solver': ['liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs']}])
    
    single_and_grid_classifier('KNN', x_train_scaled, y_train,
                               KNeighborsClassifier(n_jobs=-1),
                               [{'n_neighbors': range(1, 15)}])

    single_and_grid_classifier('SVM', x_train_scaled, y_train,
                               SVC(gamma='auto'),
                               [{
                                    'C': [0.05, 1.0, 1.5, 2.0, 3.0],
                                    'gamma': [0.2, 0.1, 0.05, 'auto_deprecated', 'scale'],
                                    'kernel': ['rbf', 'sigmoid']},
                                {
                                    'kernel': ['poly'],
                                    'degree': [3, 4, 5, 6]
                                }])

    single_and_grid_classifier('NB', x_train_scaled, y_train,
                               GaussianNB(),
                               [{}])

    single_and_grid_classifier('RandomForest', x_train_scaled, y_train,
                               RandomForestClassifier(n_jobs=-1),
                               [{'n_estimators': [5, 10, 50, 100, 200],
                                'criterion': ['gini', 'entropy']}])

    classifiers_all = [
        ('lr', LogisticRegression(solver='liblinear')),
        ('knn', KNeighborsClassifier()),
        ('svm', SVC(probability=True, gamma='auto')),
        ('nb', GaussianNB())
 #       ('rf', RandomForestClassifier())
    ]
    grid_voting_params_all = [
        {'lr__solver': ['liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs']},
        {'knn__n_neighbors': range(1, 15)},
        {
            'svm__C': [0.05, 1.0, 1.5, 2.0, 3.0],
            'svm__gamma': [0.2, 0.1, 0.05, 'auto_deprecated', 'scale'],
            'svm__kernel': ['rbf', 'sigmoid']},
        {
            'svm__kernel': ['poly'],
            'svm__degree': [3, 4, 5, 6]}
#        {
#            'rf__n_estimators': [100],
#            'rf__criterion': ['gini', 'entropy']}
    ]

    reg_score, classifier_all = grid_with_voting(classifiers_all, grid_voting_params_all, x_train_scaled, y_train)
    print(f'Grid Search With Voting classification score (all options): {reg_score}')

    preds = classifier_all.predict(x_test_scaled)

    output_preds(preds, x_test)

    '''
    grid_voting_params_specific = [
        {'lr__solver': ['liblinear']},
        {'knn__n_neighbors': [7]},
        {'svm__C': [1.5]}
        #        {'rf__criterion': ['gini'],
        #         'rf__n_estimators': [100]}
    ]

    classifiers_specific_with_params = [
        ('lr', LogisticRegression(solver='liblinear')),
        ('knn', KNeighborsClassifier(n_neighbors=7)),
        ('svm', SVC(probability=True, gamma='auto', C=1.5)),
        ('nb', GaussianNB())
        #       ('rf', RandomForestClassifier(criterion='gini', n_estimators=100))
    ]
    
    classifiers_specific = [
        ('lr', LogisticRegression(solver='liblinear')),
        ('knn', KNeighborsClassifier()),
        ('svm', SVC(probability=True, gamma='auto')),
        ('nb', GaussianNB())
        #       ('rf', RandomForestClassifier())
    ]

    reg_score, classifier_specific = grid_with_voting(classifiers_specific, grid_voting_params_specific, x_train_scaled, y_train)
    print(f'Grid Search With Voting classification score (specific options): {reg_score}')
    
    reg_score, classifier_voting = voting_only(classifiers_specific_with_params, x_train_scaled, y_train)
    print(f'Voting only classification score (specific options): {reg_score}')
'''


main()
