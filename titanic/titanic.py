import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
sns.set()


"""
TODO:
- Update titanic.MD
- Add more options of different algorithms - when have better features
- Consider adding XGBoost
- Copy from here into my summaries code that I used for the first time
"""


def read_files():
    x = pd.read_csv('input/train.csv', index_col='PassengerId')
    y = x.pop('Survived')
    x_test = pd.read_csv('input/test.csv', index_col='PassengerId')
    return x, y, x_test


def remove_outliers(x, y):
    x = x[x['Fare'] < x['Fare'].quantile(.995)]
    y = y.loc[x.index]
    return x, y


def get_title(full_name):
    return full_name.split(',')[1].split('.')[0].strip()


def handle_age(x_train, x_test):

    # 'Ms' appears only once in test, so replace it with Mrs since it's basically same ages
    x_test.loc[(x_test['Age'].isnull()) & (x_test['Name'].apply(get_title) == 'Ms'), 'Name'] = "O'Donoghue, Mrs. Bridget"

    x_train_title = x_train['Name'].apply(get_title)
    x_test_title = x_test['Name'].apply(get_title)

    for title in ['Mr', 'Miss', 'Mrs']:
        for cl in [1, 2, 3]:
            average = x_train[(x_train['Age'].isnull() == False) &
                              (x_train_title == title) &
                              (x_train['Pclass'] == cl)]['Age'].mean()
            print(f"YK: Replacing title {title} in class {cl} age with {average}")
            x_train.loc[(x_train['Age'].isnull()) &
                        (x_train_title == title) &
                        (x_train['Pclass'] == cl), 'Age'] = average
            x_test.loc[(x_test['Age'].isnull()) &
                       (x_test_title == title) &
                       (x_test['Pclass'] == cl), 'Age'] = average

    for title in ['Master', 'Dr']:
        average = x_train[(x_train['Age'].isnull() == False) & (x_train_title == title)]['Age'].mean()
        print(f"YK: Replacing title {title} age with {average}")
        x_train.loc[(x_train['Age'].isnull()) & (x_train_title == title), 'Age'] = average

        if x_test.loc[(x_test['Age'].isnull()) & (x_test_title == title), 'Age'].shape[0] > 0:
            x_test.loc[(x_test['Age'].isnull()) & (x_test_title == title), 'Age'] = average


def clean_handle_missing_categorical(x, columns_to_drop, mean_class_3_fare, min_fare, max_reasonable_fare):
    print(f'YK: Features before dropping: {x.columns.values}')

    x.drop(columns_to_drop, axis=1, inplace=True)

    print(f'YK: Features after dropping: {x.columns.values}')

    # TODO - should keep?
    # Create a new feature of number of relatives regardless of who they are
    '''
    if 'SibSp' not in columns_to_drop and 'Parch' not in columns_to_drop:
        x['Family_Num'] = x['SibSp'] + x['Parch']
        x.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    '''

    # Split 3 categorical unique values (1, 2, 3) of Pclass into 2 dummy variables for classes 1 & 2
    if 'Pclass' not in columns_to_drop:
        x['pclass_1'] = x['Pclass'].apply(lambda cl: 1 if cl == 1 else 0)
        x['pclass_2'] = x['Pclass'].apply(lambda cl: 1 if cl == 2 else 0)
        x.drop('Pclass', axis=1, inplace=True)

    # Change categorical feature 'Sex' to be 1 encoded 'Male', 1 = Male, 0 = Female
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

    if 'Fare' not in columns_to_drop:
        x['Fare'].replace({np.NaN: mean_class_3_fare}, inplace=True)

        # TODO - keep?
        # Separately removing crazy outliers
#       x['Fare'].replace({512.329200: max_reasonable_fare}, inplace=True)

        # TODO - keep?
        #x['Fare log'] = np.log(x['Fare'])
        #x['Fare log'].replace({np.NINF: min_fare}, inplace=True)
        #x.drop('Fare', axis=1, inplace=True)



def scale_train(x_train):
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler


def output_preds(preds, x_test, name_suffix):
    pred_df = pd.DataFrame(preds, index=x_test.index, columns=['Survived'])
    pred_df.to_csv(f'output/preds_{name_suffix}.csv')


def cross_valid(classifier, x_train, y_train):
    accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
    return accuracies.mean(), accuracies.std()


def fit_single_classifier(classifier, x_train, y_train):
    classifier = classifier
    classifier.fit(x_train, y_train)
    return classifier.score(x_train, y_train), classifier


def grid_search(classifier, param_grid, x_train, y_train):
    grid = GridSearchCV(classifier, param_grid, verbose=1, cv=10, n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid.score(x_train, y_train), grid


def single_and_grid_classifier(name_str, x_train, y_train, single_classifier, grid_params):
    reg_score_def, reg_std_def = cross_valid(single_classifier, x_train, y_train)

    reg_score_grid, classifier = grid_search(single_classifier, grid_params, x_train, y_train)
    reg_score_cross, reg_std_cross = cross_valid(classifier.best_estimator_, x_train, y_train)

    print(f'{name_str.ljust(20)} - Stats: Default params cross: '
          f'{round(reg_score_def, 3)} (+-{round(reg_std_def, 3)}={round(reg_score_def - reg_std_def, 3)}), '
          f'grid train: {round(reg_score_grid, 3)}, '
          f'best classifier cross: {round(reg_score_cross, 3)} '
          f'(+-{round(reg_std_cross, 3)}={round(reg_score_cross - reg_std_cross, 3)}), '
          f'best classifier:\n{classifier.best_estimator_}')
    return classifier


def grid_with_voting(classifiers, param_grid, x_train, y_train, x_test_local, y_test_local):
    voting_classifier = VotingClassifier(estimators=classifiers, voting='soft', n_jobs=-1)

    grid = GridSearchCV(voting_classifier, param_grid, verbose=1, cv=10, n_jobs=-1)
    reg_score, reg_std = cross_valid(grid, x_train, y_train)
    grid.fit(x_train, y_train)
    test_score = grid.score(x_test_local, y_test_local)
    print(f'FINAL'.ljust(44) + ' - Stats: Grid + Voting train: {round(grid.score(x_train, y_train), 3)}, '
          f'test: {round(test_score, 3)}, '
          f'best classifier cross: {round(reg_score, 3)} (+-{round(reg_std, 3)}={round(reg_score - reg_std, 3)}), '
          f'min (test/cross): {round(min(reg_score, test_score), 3)}, '
          f'best classifier:\n{grid.best_estimator_}')
    return grid


def voting_only(classifiers, x_train, y_train, weights=None):
    voting_classifier = VotingClassifier(estimators=classifiers, voting='soft', n_jobs=-1, weights=weights)
    voting_classifier.fit(x_train, y_train)
    reg_score, reg_std = cross_valid(voting_classifier, x_train, y_train)
    print(f'FINAL'.ljust(44) + ' - Stats: Best classifiers + Voting train: '
          f'{round(voting_classifier.score(x_train, y_train), 3)}, '
          f'best classifier cross: {round(reg_score, 3)} (+-{round(reg_std, 3)}={round(reg_score - reg_std, 3)}), '
          f'best classifier:\n{voting_classifier}')
    return voting_classifier


def main():
    columns_to_drop = ['Name', 'Ticket', 'Cabin',  # don't make sense to add
                       'Embarked',  # doesn't help always
                       'Fare',     # doesn't help always TODO consider returning in a different way
                       # 'Sex',
                       # 'SibSp',
                       # 'Age',
                       # 'Pclass'
                       #'Parch'      # doesn't help at the end - border line
                       ]

    x_train, y_train, x_test = read_files()

    # Shuffling doesn't seem to help.  Remove for now TODO - should stay?
    # x_train, y_train = shuffle(x_train, y_train, random_state=42)

    # TODO keep?
    # x, y = remove_outliers(x, y)

    handle_age(x_train, x_test)

    mean_class_3_fare = x_train[(x_train['Pclass'] == 1) | (x_train['Pclass'] == 2)]['Fare'].mean()
    min_fare = x_train[x_train['Fare'] > 0]['Fare'].min()
    max_reasonable_fare = x_train[x_train['Fare'] <300]['Fare'].max()

    print(f'YK: Constants: mean_class_3_fare: {mean_class_3_fare}, '
          f'min_fare: {min_fare}, max_reasonable_fare: {max_reasonable_fare}')

    clean_handle_missing_categorical(x_train, columns_to_drop, mean_class_3_fare, min_fare, max_reasonable_fare)
    clean_handle_missing_categorical(x_test, columns_to_drop, mean_class_3_fare, min_fare, max_reasonable_fare)

    scaler = scale_train(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    class_log = single_and_grid_classifier('Logistic - liblinear', x_train_scaled, y_train,
                                           LogisticRegression(solver='liblinear', n_jobs=-1),
                                           [{'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']}]) # TODO was empty
    # Following was tried also for Logistic and didn't make it much better: solver=['lbfgs', 'newton-cg', 'sag', 'saga']

    class_knn = single_and_grid_classifier('KNN - 14', x_train_scaled, y_train,
                               KNeighborsClassifier(n_jobs=-1, n_neighbors=14),
                               [{'n_neighbors': range(5, 25)}])  # # TODO was empty
    # Following was tried also and found 14 to be best: n_neighbors=[range(1, 25), 5 (lower scores), 25 (lower scored)]

    class_svm_rbf = single_and_grid_classifier('SVM - rbf', x_train_scaled, y_train,
                               SVC(gamma='auto', kernel='rbf', probability=True),
                               [{
                                   'C': [0.5, 1.0, 1.5, 2.0],
                                   'gamma': [0.2, 0.1, 0.05, 0.01, 'auto_deprecated', 'scale']}
                                ])
    class_svm_poly = single_and_grid_classifier('SVM - poly', x_train_scaled, y_train,
                               SVC(gamma='auto', kernel='poly', probability=True),
                               [{
                                   'C': [0.5, 1.0, 1.5, 2.0],
                                   'gamma': [0.2, 0.1, 0.05, 0.01, 'auto_deprecated', 'scale']}
                                ])
    # sigmoid kerned was also tried for SVM, but gave worse results

    class_nb = single_and_grid_classifier('NB', x_train_scaled, y_train,
                               GaussianNB(),
                               [{}])

    class_rf = single_and_grid_classifier('RandomForest - 9', x_train_scaled, y_train,
                               RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=9),
                               [{}])
    # TODO keep?
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
        ('lr', class_log),
        ('knn', class_knn),
        ('svm - rbf', class_svm_rbf),
        ('svm - poly', class_svm_poly),
        ('nb', class_nb),
        ('rf', class_rf)
    ]

    classifier_voting = voting_only(classifiers_specific_with_params,
                                    x_train_scaled, y_train)
                                    # TODO - should stay?
                                    # Give weights not to give rf too much weight, since it overfits
                                    # [1, 1, 1, 1, 1, 0.3])

    preds = classifier_voting.predict(x_test_scaled)
    output_preds(preds, x_test, 'best')


main()
