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
- How come 13 Fare bins? Is that based on exploratory analysis? What were we looking for to decide on this number?
- Why bin age? Because of spikes in survival rate?
- How come binning Age into 10 bins? 
- Why could it be helpful to have grouped 'Title'? There are a lot of very small categories, 
    and grouping together doesn't add new info, since already know male/female, age etc.  
    Also, Mr and Dr/Military/Noble/Clergy don't have significantly different behavior.
- How come didn't remove reference category for one-hot encoded categories?
- How come Survival_rate came out not so important in the model, for me it was 65%?
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


def extract_lastname(full_name):
    return full_name.split(',')[0]


def prepare_age_title_is_married(x):
    # TODO features - in https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic
    #   did something a bit different, based on Sex and Class, and not on title.  I think doing it on title is
    #   more exact, especially regarding differences between Mrs. and Miss

    # 'Ms' appears only once in test, so replace it with Mrs since it's basically same ages
    x.loc[(x['Age'].isnull()) & (x['Name'].apply(get_title) == 'Ms'), 'Name'] = "O'Donoghue, Mrs. Bridget"

    titles_col = x['Name'].apply(get_title)

    for title in ['Mr', 'Miss', 'Mrs']:
        for cl in [1, 2, 3]:
            average = x[(x['Age'].isnull() == False) &
                        (titles_col == title) &
                        (x['Pclass'] == cl)]['Age'].median()
            print(f"YK: Replacing title {title} in class {cl} age with {average}")
            x.loc[(x['Age'].isnull()) &
                  (titles_col == title) &
                  (x['Pclass'] == cl), 'Age'] = average

    # not enough instances of 'Master' and 'Dr' to take median by class also
    for title in ['Master', 'Dr']:
        average = x[(x['Age'].isnull() == False) & (titles_col == title)]['Age'].median()
        print(f"YK: Replacing title {title} age with {average}")
        x.loc[(x['Age'].isnull()) & (titles_col == title), 'Age'] = average

    # TODO features - consider not binning, or binning into a different number
    #   try without binning, or binning into a different number of bins
    #   make things worse, removing for now
    # x['Age'] = LabelEncoder().fit_transform(pd.qcut(x['Age'], 11))

    x['Lady married'] = titles_col.apply(lambda title: 1 if title == 'Mrs' else 0)

    # TODO features - https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic
    #   also saved title with grouping, but it doesn't seem to be helpful since
    #   there are a lot of very small categories, and grouping together doesn't add new info, since already know
    #   male/female, age etc.  Also, Mr and Dr/Military/Noble/Clergy don't have significantly different behavior
    '''
    x['Title'] = titles_col.replace(['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'],
                                              'Miss/Mrs/Ms')
    x['Title'] = titles_col.replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'],
                                              'Dr/Military/Noble/Clergy')
    '''


def prepare_family_ticket_frequencies(x):
    # add temporary last name who it's going to be decided by if it's the same family
    x['Last name'] = x['Name'].apply(extract_lastname)

    x['Known family survived %'] = np.NaN
    x['Known ticket survived %'] = np.NaN
    x['Known family/ticket survived %'] = np.NaN
    x['Family/ticket survival known'] = 1

    # to mark same information in test, need to known what last names and ticket names were known
    known_last_names = x[x['Survived'] != 2]['Last name'].unique()
    known_tickets = x[x['Survived'] != 2]['Ticket'].unique()
    mean_survival_rate = x[x['Survived'] != 2]['Survived'].mean()

    # go over all test passengers, and fill in the survival information
    for i in x.index:
        is_train = 1 if x.loc[i, 'Survived'] != 2 else 0
        did_survive = 1 if x.loc[i, 'Survived'] == 1 else 0
        last_name = x.loc[i, 'Last name']
        ticket = x.loc[i, 'Ticket']

        # if have other passengers in training set of same family whose survival information is known, copy average here
        if x[(x['Survived'] != 2) & (x['Last name'] == last_name)]['Survived'].count() > is_train:
            x.loc[i, 'Known family survived %'] = \
                (x[(x['Survived'] != 2) & (x['Last name'] == last_name)]['Survived'].sum() - did_survive) / \
                (x[(x['Survived'] != 2) & (x['Last name'] == last_name)]['Survived'].count() - is_train)

        # if have other passengers in training set of same family whose survival information is known, copy average here
        # add information for training only of how many of known survived in the same ticket
        if x[(x['Survived'] != 2) & (x['Ticket'] == ticket)]['Survived'].count() > is_train:
            x.loc[i, 'Known family survived %'] = \
                (x[(x['Survived'] != 2) & (x['Ticket'] == ticket)]['Survived'].sum() - did_survive) / \
                (x[(x['Survived'] != 2) & (x['Ticket'] == ticket)]['Survived'].count() - is_train)

        # For final value of
        if np.isnan(x.loc[i, 'Known family survived %']) == False:
            if np.isnan(x.loc[i, 'Known ticket survived %']) == False:
                # both family and ticket survival rates known, take average
                x.loc[i, 'Known family/ticket survived %'] = \
                    (x.loc[i, 'Known family survived %'] + x.loc[i, 'Known ticket survived %']) / 2
            else:
                # only family survival known, take it
                x.loc[i, 'Known family/ticket survived %'] = x.loc[i, 'Known family survived %']
        elif np.isnan(x.loc[i, 'Known ticket survived %']) == False:
            # only ticket is known - take value from ticket
            x.loc[i, 'Known family/ticket survived %'] = x.loc[i, 'Known ticket survived %']
        else:
            # none known, set mean survival value
            x.loc[i, 'Known family/ticket survived %'] = mean_survival_rate
            x.loc[i, 'Family/ticket survival known'] = 0

    print(f'YK debug: Train survival rates: \n'
          f'{x[x["Survived"] != 2]["Known family/ticket survived %"].value_counts(dropna=False)}')
    print(f'YK debug: Test survival rates: \n'
        f'{x[x["Survived"] == 2]["Known family/ticket survived %"].value_counts(dropna=False)}')

    # drop temporary columns used
    x.drop(['Last name', 'Known family survived %', 'Known ticket survived %'], axis=1, inplace=True)


def prepare_features(x, options):
    print(f'YK: Features before adding / dropping: {x.columns.values}')

    features_no_drop_after_use = []

    # Adding Age, potentially Title, and is Married
    if 'Age' not in options['columns_to_drop']:
        prepare_age_title_is_married(x)
        features_no_drop_after_use.append('Name')

    # TODO - should keep?
    # Create a new feature of number of relatives regardless of who they are
    if 'SibSp' not in options['columns_to_drop'] and 'Parch' not in options['columns_to_drop']:
        sum_sibs_parch = x['SibSp'] + x['Parch']
        x['Small family'] = sum_sibs_parch.apply(lambda size: 1 if (size < 4) and (size > 0) else 0)
        x['Large family'] = sum_sibs_parch.apply(lambda size: 1 if (size >= 4) else 0)
        # TODO features - consider removing reference category
        x['Alone'] = sum_sibs_parch.apply(lambda size: 1 if (size == 0) else 0)
        features_no_drop_after_use.append('SibSp')
        features_no_drop_after_use.append('Parch')

    # Prepare Deck features based on first letter of Cabin, unknown Cabin becomes reference category
    if 'Cabin' not in options['columns_to_drop']:
        x['Cabin'] = x['Cabin'].fillna('')
        x['Deck_AC'] = x['Cabin'].apply(lambda cab: 1 if cab.startswith('A') or cab.startswith('C') else 0)
        x['Deck_BT'] = x['Cabin'].apply(lambda cab: 1 if cab.startswith('B') or cab.startswith('T') else 0)
        x['Deck_DE'] = x['Cabin'].apply(lambda cab: 1 if cab.startswith('D') or cab.startswith('E') else 0)
        x['Deck_FG'] = x['Cabin'].apply(lambda cab: 1 if cab.startswith('F') or cab.startswith('G') else 0)
        # TODO features - consider removing reference category
        x['Deck_Other'] = x['Cabin'].apply(lambda cab: 1 if cab == '' else 0)
        print(f'Deck_Other.value_counts:\n{x["Deck_Other"].value_counts()}')
        features_no_drop_after_use.append('Cabin')

    # Split 3 categorical unique values (1, 2, 3) of Pclass into 2 dummy variables for classes 1 & 2
    if 'Pclass' not in options['columns_to_drop']:
        x['pclass_1'] = x['Pclass'].apply(lambda cl: 1 if cl == 1 else 0)
        x['pclass_2'] = x['Pclass'].apply(lambda cl: 1 if cl == 2 else 0)
        # TODO features - consider removing reference category
        x['pclass_3'] = x['Pclass'].apply(lambda cl: 1 if cl == 3 else 0)
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
        # TODO features - consider removing reference category
        x['Embarked_C'] = x['Embarked'].map({np.NaN: 0, 'S': 0, 'C': 1, 'Q': 0})
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

    if 'Ticket' not in options['columns_to_drop']:
        x['Ticket_Frequency'] = x.groupby('Ticket')['Ticket'].transform('count')
        features_no_drop_after_use.append('Ticket')

    prepare_family_ticket_frequencies(x)

    x.drop(options['columns_to_drop'], axis=1, inplace=True)
    x.drop(features_no_drop_after_use, axis=1, inplace=True)

    print(f'YK: Features after dropping: {x.columns.values}')
    print(f'YK: both.info():\n{x.info()}')
    print(f'YK: Value counts of all values:')
    for feat in x.columns.values:
        print(f'--------------- {feat}:')
        print(x[feat].value_counts())


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
    start_time = time.time()

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
          f'time took: {round(time.time() - start_time)} sec, '
          f'best classifier:\n{classifier.best_estimator_}')
    classifier.best_estimator_.fit(x_train, y_train)
    return classifier.best_estimator_
    #classifier.fit(x_train, y_train)
    #return classifier


# TODO should leave - if leaving need to change a whole bunch of things, was not updated
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
    start_time = time.time()
    voting_classifier = VotingClassifier(estimators=classifiers, voting='soft', n_jobs=-1, weights=weights)
    reg_score, reg_std = cross_valid(voting_classifier, x_train, y_train)
    voting_classifier.fit(x_train, y_train)
    print(f'FINAL'.ljust(44) + ' - Stats: Best classifiers + Voting train: '
          f'{round(voting_classifier.score(x_train, y_train), 3)}, '
          f'best classifier cross: {round(reg_score, 3)} (+-{round(reg_std, 3)}={round(reg_score - reg_std, 3)}), '
          f'time took: {round(time.time() - start_time)} sec, '
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

    class_rf_5 = single_and_grid_classifier('RandomForest - 5', x_train_scaled, y_train,
                                          RandomForestClassifier(n_jobs=-1, n_estimators=1000, max_depth=5),
                                          options,
                                          [{}])

    class_rf_4 = single_and_grid_classifier('RandomForest - 4', x_train_scaled, y_train,
                                          RandomForestClassifier(n_jobs=-1, n_estimators=1000, max_depth=4),
                                          options,
                                          [{}])

    class_rf_3 = single_and_grid_classifier('RandomForest - 3', x_train_scaled, y_train,
                                            RandomForestClassifier(n_jobs=-1, n_estimators=1000, max_depth=3),
                                            options,
                                            [{}])

    start_time = time.time()

    class_rf_exp = RandomForestClassifier(n_estimators=1000,
                                           max_depth=5,
                                           #min_samples_split=3,
                                           #min_samples_leaf=5,
                                           n_jobs=-1)
    class_rf_exp.fit(x_train_scaled, y_train)
    reg_score = class_rf_exp.score(x_train_scaled, y_train)
    reg_score_cross, reg_std_cross = cross_valid(class_rf_exp, x_train_scaled, y_train)

    importances = pd.DataFrame({'Importance': class_rf_exp.feature_importances_}, index=x_train.columns)
    print(f'{"RandomForest Explicit 5".ljust(20)} - Stats: Default params cross: '
          f'grid train: {round(reg_score, 3)}, '
          f'best classifier cross: {round(reg_score_cross, 3)} '
          f'(+-{round(reg_std_cross, 3)}={round(reg_score_cross - reg_std_cross, 3)}), '
          f'time took: {round(time.time() - start_time)} sec = {round((time.time() - start_time) / 60)} min, '
          f'importances:\n{importances["Importance"].sort_values()}')


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
        ('rf_5', class_rf_5),
        ('rf_4', class_rf_4),
        ('rf_3', class_rf_3),
        ('xgb', class_xgb)
    ]

    classifier_voting = voting_only(classifiers_specific_with_params,
                                    x_train_scaled, y_train)
                                    # TODO - should stay?
                                    # Give weights not to give rf too much weight, since it overfits
                                    # [1, 1, 1, 1, 1, 0.3])

    preds = classifier_voting.predict(x_test_scaled)
    output_preds(preds, x_test, 'best')

    preds = class_rf_exp.predict(x_test_scaled)
    output_preds(preds, x_test, 'rf_explicit')

    preds = class_xgb.predict(x_test_scaled)
    output_preds(preds, x_test, 'xgb')

    print(f'YK: Time took: {time.time() - start_time} seconds = {round((time.time() - start_time)/60)} minutes ')


options = {
    'columns_to_drop': [# 'Embarked',  # doesn't help always
                        # 'Ticket' - used for ticket frequency
                        # 'Name' - used for title
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
