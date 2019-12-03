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
from sklearn.ensemble import RandomForestRegressor
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

NUM_TRAIN_SAMPLES = 891

def read_files():
    train = pd.read_csv('input/train.csv', index_col='PassengerId')
    x_test = pd.read_csv('input/test.csv', index_col='PassengerId')
    return train, x_test


# TODO go over all combine and split, remove not used
def combine_train_x_test(train, x_test):
    return pd.concat([train.drop('Survived', axis=1), x_test])


def combine_x_train_x_test(x_train, x_test):
    return pd.concat([x_train, x_test])


def combine_x_train_y_train(x_train, y_train):
    train = x_train.copy()
    train['Survived'] = y_train
    return train


def split_into_x_train_x_test(both):
    return both.iloc[:NUM_TRAIN_SAMPLES], both.iloc[NUM_TRAIN_SAMPLES:]


def split_into_x_train_y_train(train):
    return train.drop('Survived', axis=1), train['Survived']


def get_title(full_name):
    return full_name.split(',')[1].split('.')[0].strip()


def extract_lastname(full_name):
    return full_name.split(',')[0]


def impute_age_regression(both):
    # Conclusions:
    # - Sex doesn't seem to help (because there is title)
    # - Didn't try, harder to add, and probably won't help much: 'Fare bin', 'Deck' has size issues
    features_base_age_on = ['Pclass', 'Title', 'Parch', 'SibSp', 'Ticket_Frequency', 'Embarked']
    x_test = both.loc[both['Age'].isnull(), features_base_age_on]
    x_train = both.loc[both['Age'].isnull() == False, features_base_age_on]
    y_train = both.loc[both['Age'].isnull() == False, 'Age']

    for cat in ['Title', 'Embarked']:
        x_test[cat] = LabelEncoder().fit_transform(x_test[cat])
        x_train[cat] = LabelEncoder().fit_transform(x_train[cat])

    age_model = RandomForestRegressor(n_estimators=1000)
    age_model.fit(x_train, y_train)
    print(f"Age prediction feature importance, features: {features_base_age_on}, importance:\n{age_model.feature_importances_}")
    age_preds = age_model.predict(x_test)

    both.loc[both['Age'].isnull(), 'Age'] = age_preds

    print(f'YK: Age prediction score: {round(age_model.score(x_train, y_train) * 100,1)}')
    return age_preds


# TODO decide if leaving
def impute_age_by_title_pclass(both):
    for title in ['Mr', 'Miss', 'Mrs']:
        for cl in [1, 2, 3]:
            average = both[(both['Age'].isnull() == False) &
                           (both['Title'] == title) &
                           (both['Pclass'] == cl)]['Age'].median()
            print(f"YK: Replacing title {title} in class {cl} age ("
                  f"{both.loc[(both['Age'].isnull()) & (both['Title'] == title) & (both['Pclass'] == cl), 'Age'].shape[0]}"
                  f" observations) with {average}")
            both.loc[(both['Age'].isnull()) &
                     (both['Title'] == title) &
                     (both['Pclass'] == cl), 'Age'] = average

    # not enough instances of 'Master' to take median by class also
    title = 'Master'
    average = both[(both['Age'].isnull() == False) & (both['Title'] == title)]['Age'].median()
    print(f"YK: Replacing title {title} age ("
          f"{both.loc[(both['Age'].isnull()) & (both['Title'] == title), 'Age'].shape[0]}"
          f" observations) with {average}")
    both.loc[(both['Age'].isnull()) & (both['Title'] == title), 'Age'] = average


def prepare_family_ticket_frequencies(both, y_train):

    max_train_index = y_train.index[-1]

    train = combine_x_train_y_train(split_into_x_train_x_test(both)[0], y_train)

    # add temporary last name who it's going to be decided by if it's the same family
    both['Last name'] = both['Name'].apply(extract_lastname)

    both['Known family survived %'] = np.NaN
    both['Known ticket survived %'] = np.NaN
    both['Known family/ticket survived %'] = np.NaN
    both['Family/ticket survival known'] = 1

    # go over all test passengers, and fill in the survival information
    for i in both.index:
        is_train = 1 if i <= max_train_index else 0
        did_survive = 1 if (is_train == 1) and (train.loc[i, 'Survived'] == 1) else 0
        last_name = both.loc[i, 'Last name']
        ticket = both.loc[i, 'Ticket']

        # if have other passengers in training set of same family whose survival information is known, copy average here
        if train[both['Last name'] == last_name]['Survived'].count() > is_train:
            both.loc[i, 'Known family survived %'] = \
                (train[both['Last name'] == last_name]['Survived'].sum() - did_survive) / \
                (train[both['Last name'] == last_name]['Survived'].count() - is_train)

        # if have other passengers in training set of same family whose survival information is known, copy average here
        # add information for training only of how many of known survived in the same ticket
        if train[train['Ticket'] == ticket]['Survived'].count() > is_train:
            both.loc[i, 'Known family survived %'] = \
                (train[train['Ticket'] == ticket]['Survived'].sum() - did_survive) / \
                (train[train['Ticket'] == ticket]['Survived'].count() - is_train)

        # For final value of
        if np.isnan(both.loc[i, 'Known family survived %']) == False:
            if np.isnan(both.loc[i, 'Known ticket survived %']) == False:
                # both family and ticket survival rates known, take average
                both.loc[i, 'Known family/ticket survived %'] = \
                    (both.loc[i, 'Known family survived %'] + both.loc[i, 'Known ticket survived %']) / 2
            else:
                # only family survival known, take it
                both.loc[i, 'Known family/ticket survived %'] = both.loc[i, 'Known family survived %']
        elif np.isnan(both.loc[i, 'Known ticket survived %']) == False:
            # only ticket is known - take value from ticket
            both.loc[i, 'Known family/ticket survived %'] = both.loc[i, 'Known ticket survived %']
        else:
            # none known, set mean survival value
            both.loc[i, 'Known family/ticket survived %'] = train['Survived'].mean()
            both.loc[i, 'Family/ticket survival known'] = 0

    print(f'YK debug: Train survival rates: \n'
          f'{split_into_x_train_x_test(both)[0]["Known family/ticket survived %"].value_counts(dropna=False)}')
    print(f'YK debug: Test survival rates: \n'
        f'{split_into_x_train_x_test(both)[1]["Known family/ticket survived %"].value_counts(dropna=False)}')

    # drop temporary columns used
    both.drop(['Last name', 'Known family survived %', 'Known ticket survived %'], axis=1, inplace=True)


# TODO decide if to keep
def manual_fare_bin_numeric(fare):
    if fare == 0:
        return 0  # based on kde
    elif fare < 4:
        return 1  # based on kde
    elif fare < 5:
        return 2  # based on kde
    elif fare < 7:
        return 3  # based on kde
    elif fare <= 7.796:
        return 4  # based on survival rate
    elif fare <= 7.896:
        return 5  # based on survival rate
    elif fare <= 7.925:
        return 6  # based on survival rate
    elif fare <= 8.662:
        return 7  # based on survival rate and KDE
    elif fare <= 12.5:
        return 8  # based on KDE
    elif fare <= 13.5:
        return 9  # based on survival rate and KDE
    else:
        return 10


def manual_fare_bin_categorical(fare):
    if fare == 0:
        return '0'  # based on kde
    elif fare < 4:
        return '0.1-4'  # based on kde
    elif fare < 5:
        return '4-5'  # based on kde
    elif fare < 7:
        return '5-7'  # based on kde
    elif fare <= 7.796:
        return '7-7.796'  # based on survival rate
    elif fare <= 7.896:
        return '7.796-7.896'  # based on survival rate
    elif fare <= 7.925:
        return '7.896-7.925'  # based on survival rate
    elif fare <= 8.662:
        return '7.925-8.662'  # based on survival rate and KDE
    elif fare <= 12.5:
        return '8.662-12.5'  # based on KDE
    elif fare <= 13.5:
        return '12.5-13.5'  # based on survival rate and KDE
    else:
        return 10


def manual_age_bin(fare):
    if fare <= 4:
        return '-4'  # based on survival rate
    elif fare <= 11:
        return '4-11'  # based on survival rate
    elif fare <= 24:
        return '11-24'  # based on survival rate
    elif fare <= 26:
        return '24-26'  # based on survival rate
    elif fare <= 27:
        return '26-27'  # based on survival rate
    elif fare <= 31:
        return '27-31'  # based on survival rate
    elif fare <= 32:
        return '31-32'  # based on survival rate
    elif fare <= 40:
        return '32-40'  # based on KDE
    elif fare <= 48:
        return '40-48'  # based on KDE
    elif fare <= 57:
        return '48-57'  # based on KDE
    else:
        return '57+'  # based on KDE



def prepare_features(train, x_test, options):
    num_train_samples = train.shape[0]

    print(f'YK debug: num_train_samples:{num_train_samples}')
    print(f'YK: Features before adding / dropping: {x_test.columns.values}')

    features_to_drop_after_use = []
    features_to_add_dummies = []
    both = combine_train_x_test(train, x_test)
    print(f'YK debug: both head:\n{both.head}')

    # 1 ---> Adding title, see details in Advanced feature engineering.ipynb
    both['Title'] = both['Name'].apply(get_title).replace(
        {'Lady': 'Mrs', 'Mme': 'Mrs', 'Dona': 'Mrs', 'the Countess': 'Mrs',
         'Ms': 'Miss', 'Mlle': 'Miss',
         'Sir': 'Mr', 'Major': 'Mr', 'Capt': 'Mr', 'Jonkheer': 'Mr', 'Don': 'Mr', 'Col': 'Mr', 'Rev': 'Mr', 'Dr': 'Mr'})
    features_to_add_dummies.append('Title')
    features_to_drop_after_use.append('Name')

    # 2 ---> Create a new feature of number 'Family size' of relatives regardless of who they are
    #   Group SibSp, Parch, Family size based on different survival rates
    both['Family size'] = 1 + both['SibSp'] + both['Parch']
    both['SibSpBin'] = both['SibSp'].replace({0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                                           5: '5+', 8: '5+'})
    both['ParchBin'] = both['Parch'].replace({0: '0',
                                           1: '123', 2: '123', 3: '123',
                                           4: '4+', 5: '4+', 6: '4+', 9: '4+'})
    both['Family size'] = both['Family size'].replace({1: '1',
                                                       2: '23', 3: '23',
                                                       4: '4',
                                                       5: '567', 6: '567', 7: '567',
                                                       8: '8+', 11: '8+'})
    features_to_add_dummies.append('SibSpBin')
    features_to_add_dummies.append('ParchBin')
    features_to_add_dummies.append('Family size')
    features_to_drop_after_use.append('SibSp')
    features_to_drop_after_use.append('Parch')


    # 3. ----> Prepare Deck features based on first letter of Cabin, unknown Cabin becomes reference category
    # Reference category is being removed later based on importance of each of the categories
    both['Cabin'] = both['Cabin'].fillna('unknown')
    both['Deck'] = both['Cabin'].apply(lambda cab: cab[0] if (cab != 'unknown') else cab)
    both['DeckBin'] = both['Deck'].replace({'unknown': 'unknown_T', 'T': 'unknown_T',
                                                'B': 'BDE', 'D': 'BDE', 'E': 'BDE',
                                                'C': 'CF', 'F': 'CF',
                                                'A': 'AG', 'G': 'AG'})
    features_to_drop_after_use.append('Cabin')
    features_to_drop_after_use.append('Deck')
    features_to_add_dummies.append('DeckBin')


    # 4 ---> Add Pclass category
    features_to_add_dummies.append('Pclass')

    # 5 ---> Add Sex
    both['Sex'] = both['Sex'].map({'male': 1, 'female': 0})

    # 6 ---> Add Embarked, fill the 2 missing values with the most common S
    both['Embarked'] = both['Embarked'].fillna('S')
    features_to_add_dummies.append('Embarked')

    # 7 --> Add new feature of Ticked Frequency - how many times this ticket appeared,
    #   kind of size of family but of ticket
    both['Ticket_Frequency'] = both.groupby('Ticket')['Ticket'].transform('count')
    features_to_drop_after_use.append('Ticket')

    # 8 --> Fare. Add new category of "Fare per person" since fares are currently per ticket, and Set missing values
    #   Replace with bins, and get rid of regular Fare
    both['Fare per person'] = both['Fare'] / both['Ticket_Frequency']
    # In the same class, Fare per person has a tight distribution, so just take median
    both['Fare per person'] = both['Fare per person'].fillna(both[both['Pclass'] == 3]['Fare per person'].median())
    # Since the missing Fare is only of a person with Ticket_Frequency 1, take median of class 3 of Fare per person
    both['Fare'] = both['Fare'].fillna(both[both['Pclass'] == 3]['Fare per person'].median())
    # Add categorical category of manual bins of fares (see Advanced feature engineering.ipynb notebook)
    # TODO decide if to leave 'Fare bin' categorical, or turn into numerical
    both['Fare bin'] = both['Fare per person'].apply(manual_fare_bin_categorical)
    features_to_drop_after_use.append('Fare')
    features_to_drop_after_use.append('Fare per person')
    features_to_add_dummies.append('Fare bin')
    # TODO if deleting manual bin, add additional category of free tickets and Fare per person of between 0 and 4.5,
    #   and above 13.5 that clearly affect the results

    # 9 --> Add frequencies of survival per family (based on last name) and ticket
    prepare_family_ticket_frequencies(both, train['Survived'])

    # 10 --> Age - fill in missing values, bin
    impute_age_regression(both)
    #impute_age_by_title_pclass(both)
    both['Age'] = both['Age'].apply(manual_age_bin)
    print(f"Age value_counts:\n{both['Age'].value_counts().sort_index()}")
    features_to_add_dummies.append('Age')

    both.drop(options['columns_to_drop'], axis=1, inplace=True)
    both.drop(features_to_drop_after_use, axis=1, inplace=True)

    print(f'YK: Features after dropping before adding dummy, shape {both.shape}: {both.columns.values}')

    both = pd.get_dummies(both, columns=features_to_add_dummies)

    print(f'YK: Features after dropping after adding dummy, shape {both.shape}: {both.columns.values}')

    print(f'YK: both.info():\n{both.info()}')
    print(f'YK: Value counts of all values:')
    for feat in both.columns.values:
        print(f'--------------- {feat}:')
        print(both[feat].value_counts())

    both.corr().to_csv('output/feature_correlations.csv')

    return split_into_x_train_x_test(both)


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
    start_time_total = time.time()

    # TODO remove all the compare lines
    train, x_test = read_files()
    debug_train_index_before = train.index
    debug_test_index_before = x_test.index

    x_train, x_test = prepare_features(train, x_test, options)

    debug_x_train_index_after = x_train.index
    debug_test_index_after = x_test.index
    assert(np.array_equal(debug_train_index_before, debug_x_train_index_after))
    assert (np.array_equal(debug_test_index_before, debug_test_index_after))

    y_train = train['Survived']

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

    start_time = time.time()

    class_rf_exp_5 = RandomForestClassifier(n_estimators=1000,  # also tried 2000
                                           max_depth=5,  # also tried leader 7, but had worse performance
                                           # min_samples_split=6,
                                           # min_samples_leaf=6,
                                           # oob_score=True,
                                           n_jobs=-1)
    class_rf_exp_5.fit(x_train_scaled, y_train)
    reg_score = class_rf_exp_5.score(x_train_scaled, y_train)
    reg_score_cross, reg_std_cross = cross_valid(class_rf_exp_5, x_train_scaled, y_train)

    importances = pd.DataFrame({'Importance': class_rf_exp_5.feature_importances_}, index=x_train.columns)
    print(f'{"RandomForest Explicit 5".ljust(20)} - Stats: Default params cross: '
          f'grid train: {round(reg_score, 3)}, '
          f'best classifier cross: {round(reg_score_cross, 3)} '
          f'(+-{round(reg_std_cross, 3)}={round(reg_score_cross - reg_std_cross, 3)}), '
          f'time took: {round(time.time() - start_time)} sec = {round((time.time() - start_time) / 60)} min, '
          f'importances:\n{importances["Importance"].sort_values()}')


    class_xgb = single_and_grid_classifier('XGB', x_train_scaled, y_train,
                                           xgb.XGBClassifier(objective='binary:logistic', random_state=42, n_jobs=-1,
                                                             n_estimators=1000),
                                                             # , max_depth=4),
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
        ('xgb', class_xgb)
    ]

    classifier_voting = voting_only(classifiers_specific_with_params,
                                    x_train_scaled, y_train)
                                    # TODO - should stay?
                                    # Give weights not to give rf too much weight, since it overfits
                                    # [1, 1, 1, 1, 1, 0.3])

    preds = classifier_voting.predict(x_test_scaled)
    output_preds(preds, x_test, 'best')

    preds = class_svm_rbf.predict(x_test_scaled)
    output_preds(preds, x_test, 'svm_rbf')

    preds = class_rf_exp_5.predict(x_test_scaled)
    output_preds(preds, x_test, 'rf_5_explicit')

    preds = class_xgb.predict(x_test_scaled)
    output_preds(preds, x_test, 'xgb')

    print(f'YK: Time took: {time.time() - start_time_total} seconds = '
          f'{round((time.time() - start_time_total)/60)} minutes ')


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

                        #'pclass_2',
                        #'Alone',
                        #'Deck_FG',
                        #'Deck_AC',
                        #'Deck_BT',
                        #'Embarked',

                        # 'Family/ticket survival known' - removing it gives slightly worse results and larger deltas
                        ],
    'hyperparams_optimization': False

}

main(options)
