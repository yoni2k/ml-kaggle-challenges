import warnings
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
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
import xgboost as xgb
sns.set()


"""
TODO:
- Update titanic.MD
- Add more options of different algorithms - when have better features
- Consider adding XGBoost
- Copy from here into my summaries code that I used for the first time
"""


# TODO: ask Notbook owner of https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic
'''
- How come age is replaced with Median and not with Mean?
- How come fare is replaced with Median and not with mean? - Since only one case will not make a difference
- Why bin fare? To get rid of outliers or some other reason?
# How come 13 Fare bins? Is that based on exploratory analysis? What were we looking for to decide on this number?
- Why bin age? Because of spikes in survival rate?
'''


def read_files():
    train = pd.read_csv('input/train.csv', index_col='PassengerId')
    x_test = pd.read_csv('input/test.csv', index_col='PassengerId')
    return train, x_test


# TODO keep?
def remove_outliers(x, y):
    x = x[x['Fare'] < x['Fare'].quantile(.995)]
    y = y.loc[x.index]
    return x, y


def get_title(full_name):
    return full_name.split(',')[1].split('.')[0].strip()


def handle_age(both):

    # TODO features - in https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic
    #   did something a bit different, based on Sex and Class, and not on title.  I think doing it on title is
    #   more exact, especially regarding differences between Mrs. and Miss

    # 'Ms' appears only once in test, so replace it with Mrs since it's basically same ages
    both.loc[(both['Age'].isnull()) & (both['Name'].apply(get_title) == 'Ms'), 'Name'] = "O'Donoghue, Mrs. Bridget"

    both_title = both['Name'].apply(get_title)

    for title in ['Mr', 'Miss', 'Mrs']:
        for cl in [1, 2, 3]:
            average = both[(both['Age'].isnull() == False) &
                           (both_title == title) &
                           (both['Pclass'] == cl)]['Age'].median()
            print(f"YK: Replacing title {title} in class {cl} age with {average}")
            both.loc[(both['Age'].isnull()) &
                     (both_title == title) &
                     (both['Pclass'] == cl), 'Age'] = average

    # not enough instances of 'Master' and 'Dr' to take median by class also
    for title in ['Master', 'Dr']:
        average = both[(both['Age'].isnull() == False) & (both_title == title)]['Age'].median()
        print(f"YK: Replacing title {title} age with {average}")
        both.loc[(both['Age'].isnull()) & (both_title == title), 'Age'] = average

    # TODO features - consider not binning, or binning into a different number
    #    both['Age'] = StandardScaler().fit_transform(pd.qcut(both['Age'], 10))


def prepare_features(x, options):
    print(f'YK: Features before adding / dropping: {x.columns.values}')

    features_no_drop_after_use = []

    if 'Age' not in options['columns_to_drop']:
        handle_age(x)

    # TODO - should keep?
    # Create a new feature of number of relatives regardless of who they are
    '''
    if 'SibSp' not in columns_to_drop and 'Parch' not in columns_to_drop:
        x['Family_Num'] = x['SibSp'] + x['Parch']
        x.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    '''

    # TODO features - return
    if 'Cabin' not in options['columns_to_drop']:
        x['Cabin'] = x['Cabin'].fillna('')
        x['Deck_AC'] = x['Cabin'].apply(lambda cab: 1 if cab.startswith('A') or cab.startswith('C') else 0)
        x['Deck_BT'] = x['Cabin'].apply(lambda cab: 1 if cab.startswith('B') or cab.startswith('T') else 0)
        x['Deck_DE'] = x['Cabin'].apply(lambda cab: 1 if cab.startswith('D') or cab.startswith('E') else 0)
        x['Deck_FG'] = x['Cabin'].apply(lambda cab: 1 if cab.startswith('F') or cab.startswith('G') else 0)
        print(f"YK: x['Deck_AC'].sum(): {x['Deck_AC'].sum()}")
        print(f"YK: x['Deck_BT'].sum(): {x['Deck_BT'].sum()}")
        print(f"YK: x['Deck_DE'].sum(): {x['Deck_DE'].sum()}")
        print(f"YK: x['Deck_FG'].sum(): {x['Deck_FG'].sum()}")
        features_no_drop_after_use.append('Cabin')


    # Split 3 categorical unique values (1, 2, 3) of Pclass into 2 dummy variables for classes 1 & 2
    if 'Pclass' not in options['columns_to_drop']:
        x['pclass_1'] = x['Pclass'].apply(lambda cl: 1 if cl == 1 else 0)
        x['pclass_2'] = x['Pclass'].apply(lambda cl: 1 if cl == 2 else 0)
        features_no_drop_after_use.append('Pclass')

    # Change categorical feature 'Sex' to be 1 encoded 'Male', 1 = Male, 0 = Female
    if 'Sex' not in options['columns_to_drop']:
        x['Male'] = x['Sex'].map({'male': 1, 'female': 0})
        features_no_drop_after_use.append('Sex')

    ''' 
        Change categorical feature 'Embarked' with 3 values ('S', 'C', 'Q') to be 2 dummy variables: 
            Embarked_S, Embarked_Q with 'C' being a reference variable
            In addition, handle 2 missing values to have them the most common value 'S'
    '''
    if 'Embarked' not in options['columns_to_drop']:
        x['Embarked_S'] = x['Embarked'].map({np.NaN: 1, 'S': 1, 'C': 0, 'Q': 0})
        x['Embarked_Q'] = x['Embarked'].map({np.NaN: 0, 'S': 0, 'C': 0, 'Q': 1})
        features_no_drop_after_use.append('Embarked')

    if 'Fare' not in options['columns_to_drop']:
        # TODO feature - not really important since it's only 1, but: We can assume that Fare is related to family size (Parch and SibSp) and Pclass features.Median Fare value of a male with a third class ticket and no family is a logical choice to fill the missing value.
        ''' - Code copied from notebook online:
        med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
        # Filling the missing value in Fare with the median Fare of 3rd class alone passenger
        df_all['Fare'] = df_all['Fare'].fillna(med_fare)
        '''
        mean_class_3_fare = x[(x['Pclass'] == 3) & (x['SibSp'] == 0) & (x['Parch'] == 0)]['Fare'].median()
        x['Fare'] = x['Fare'].fillna(mean_class_3_fare)

        # TODO features - how come 13 bins? Is that based on exploratory analysis? What were we looking for?
        #   Try as optimization input with a different number of bins
        x['Fare'] = LabelEncoder().fit_transform(pd.qcut(x['Fare'], 13))

    x.drop(options['columns_to_drop'], axis=1, inplace=True)
    x.drop(features_no_drop_after_use, axis=1, inplace=True)

    print(f'YK: Features after dropping: {x.columns.values}')
    print(f'YK: both.info():\n{x.info()}')


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


def single_and_grid_classifier(name_str, x_train, y_train, single_classifier, options, grid_params):
    reg_score_def, reg_std_def = cross_valid(single_classifier, x_train, y_train)

    if not options['hyperparams_optimization']:
        grid_params = [{}]

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


def main(options):
    start_time = time.time()

    train, x_test = read_files()

    # Shuffling doesn't seem to help.  Remove for now TODO - should stay?
    # x_train, y_train = shuffle(x_train, y_train, random_state=42)

    # TODO keep?
    # x, y = remove_outliers(x, y)

    x_test['Survived'] = 2
    both = pd.concat([train, x_test], axis=0)

    prepare_features(both, options)

    x_train = both[both['Survived'] != 2].drop('Survived', axis=1)
    y_train = both[both['Survived'] != 2]['Survived']
    x_test = both[both['Survived'] == 2].drop('Survived', axis=1)

    scaler = scale_train(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    class_log = single_and_grid_classifier('Logistic - liblinear', x_train_scaled, y_train,
                                           LogisticRegression(solver='liblinear', n_jobs=-1),
                                           options,
                                           [{'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']}],) # TODO was empty
    # Following was tried also for Logistic and didn't make it much better: solver=['lbfgs', 'newton-cg', 'sag', 'saga']

    class_knn = single_and_grid_classifier('KNN - 14', x_train_scaled, y_train,
                                           KNeighborsClassifier(n_jobs=-1, n_neighbors=14),
                                           options,
                                           [{'n_neighbors': range(5, 25)}])  # # TODO was empty
    # Following was tried also and found 14 to be best: n_neighbors=[range(1, 25), 5 (lower scores), 25 (lower scored)]

    class_svm_rbf = single_and_grid_classifier('SVM - rbf', x_train_scaled, y_train,
                                               SVC(gamma='auto', kernel='rbf', probability=True),
                                               options,
                                               [{
                                                   'C': [0.5, 1.0, 1.5, 2.0],
                                                   'gamma': [0.2, 0.1, 0.05, 0.01, 'auto_deprecated', 'scale']}
                                               ])
    class_svm_poly = single_and_grid_classifier('SVM - poly', x_train_scaled, y_train,
                                                SVC(gamma='auto', kernel='poly', probability=True),
                                                options,
                                                [{
                                                    'C': [0.5, 1.0, 1.5, 2.0],
                                                    'gamma': [0.2, 0.1, 0.05, 0.01, 'auto_deprecated', 'scale']}
                                                ])
    # sigmoid kerned was also tried for SVM, but gave worse results

    class_nb = single_and_grid_classifier('NB', x_train_scaled, y_train,
                                          GaussianNB(),
                                          options,
                                          [{}])

    class_rf = single_and_grid_classifier('RandomForest - 9', x_train_scaled, y_train,
                                          RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=9),
                                          options,
                                          [{}])

    class_xgb = single_and_grid_classifier('XGB', x_train_scaled, y_train,
                                           xgb.XGBClassifier(objective='binary:logistic', random_state=42, n_jobs=-1),
                                           options,
                                           [{
                                               'max_depth': range(2, 8, 1),  # default 3
                                               # 'n_estimators': range(60, 260, 40), # default 100
                                               # 'learning_rate': [0.3, 0.2, 0.1, 0.01],  # , 0.001, 0.0001
                                               # 'min_child_weight': [0.5, 1, 2],  # default 1
                                               # 'subsample': [i / 10.0 for i in range(6, 11)], # default 1, not sure needed
                                               # 'colsample_bytree': [i / 10.0 for i in range(6, 11)] # default 1, not sure needed
                                               # 'gamma': [i / 10.0 for i in range(3)]  # default 0
                                            }])

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
        ('rf', class_rf),
        ('xgb', class_xgb)
    ]

    classifier_voting = voting_only(classifiers_specific_with_params,
                                    x_train_scaled, y_train)
                                    # TODO - should stay?
                                    # Give weights not to give rf too much weight, since it overfits
                                    # [1, 1, 1, 1, 1, 0.3])

    preds = classifier_voting.predict(x_test_scaled)
    output_preds(preds, x_test, 'best')

    preds = class_xgb.predict(x_test_scaled)
    output_preds(preds, x_test, 'xgb')

    print(f'YK: Time took: {time.time() - start_time} seconds = {round((time.time() - start_time)/60)} minutes ')


options = {
    'columns_to_drop': ['Name', 'Ticket',  # don't make sense to add
                        'Embarked',  # doesn't help always
                        # 'Fare',     # doesn't help always TODO consider returning in a different way
                        # 'Cabin' - helps if take out 'Deck' from first letter, currently doesn't make a difference to have it or not
                        # 'Sex',
                        # 'SibSp',
                        # 'Age',
                        # 'Pclass'
                        # 'Parch'      # doesn't help at the end - border line
                        ],
    'hyperparams_optimization': False

}

main(options)
