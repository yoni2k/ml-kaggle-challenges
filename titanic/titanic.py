import time
import numpy as np
import pandas as pd
import seaborn as sns

import os
import csv

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
sns.set()

NUM_TRAIN_SAMPLES = 891
RANDOM_STATE_FIRST = 50


def read_files():
    train = pd.read_csv('input/train.csv', index_col='PassengerId')
    x_test = pd.read_csv('input/test.csv', index_col='PassengerId')
    return train, x_test


def combine_train_x_test(train, x_test):
    return pd.concat([train.drop('Survived', axis=1), x_test])


def combine_x_train_y_train(x_train, y_train):
    train = x_train.copy()
    train['Survived'] = y_train
    return train


def split_into_x_train_x_test(both):
    return both.iloc[:NUM_TRAIN_SAMPLES], both.iloc[NUM_TRAIN_SAMPLES:]


def get_title(full_name):
    return full_name.split(',')[1].split('.')[0].strip()


def extract_lastname(full_name):
    return full_name.split(',')[0]


# TODO keep?
def impute_age_regression_train_test(both):
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
    print(f"Debug: Age prediction feature importance, features: {features_base_age_on}, "
          f"importance:\n{age_model.feature_importances_}")
    age_preds = age_model.predict(x_test)

    both.loc[both['Age'].isnull(), 'Age'] = age_preds

    print(f'Debug: Age prediction score: {round(age_model.score(x_train, y_train) * 100,1)}')
    return age_preds


# TODO - go over all functions, make sure 'train' use makes sense.  Any nicer way to do that?
def impute_age_regression_based_on_train(data, train):
    # Conclusions:
    # - Sex doesn't seem to help (because there is title)
    # - Didn't try, harder to add, and probably won't help much: 'Fare bin', 'Deck' has size issues
    # TODO - used to be also based on leaked feature of Ticket_Frequency
    train = train.copy()
    train['Title'] = train['Name'].apply(get_title).replace(title_map)
    features_base_age_on = ['Pclass', 'Title', 'Parch', 'SibSp', 'Embarked']
    x_test = data.loc[data['Age'].isnull(), features_base_age_on]
    x_train = train.loc[train['Age'].isnull() == False, features_base_age_on]
    y_train = train.loc[train['Age'].isnull() == False, 'Age']

    x_train['Embarked'] = x_train['Embarked'].fillna('S')

    for cat in ['Title', 'Embarked']:
        x_test[cat] = LabelEncoder().fit_transform(x_test[cat])
        print(f'Debug: Shape x_train: {x_train.shape}, columns: {x_train.columns}')
        x_train[cat] = LabelEncoder().fit_transform(x_train[cat])

    age_model = RandomForestRegressor(n_estimators=1000)
    age_model.fit(x_train, y_train)
    print(f"Debug: Age prediction feature importance, features: {features_base_age_on}, "
          f"importance:\n{age_model.feature_importances_}")
    age_preds = age_model.predict(x_test)

    data.loc[data['Age'].isnull(), 'Age'] = age_preds

    print(f'Debug: Age prediction score: {round(age_model.score(x_train, y_train) * 100, 1)}')
    return age_preds


# TODO - should keep since we removed leaked features?
def prepare_family_ticket_frequencies_leaked(both, y_train):

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
            both.loc[i, 'Known ticket survived %'] = \
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

    print(f'Debug: Train survival rates: \n'
          f'{split_into_x_train_x_test(both)[0]["Known family/ticket survived %"].value_counts(dropna=False)}')
    print(f'Debug: Test survival rates: \n'
          f'{split_into_x_train_x_test(both)[1]["Known family/ticket survived %"].value_counts(dropna=False)}')

    # drop temporary columns used
    both.drop(['Last name', 'Known family survived %', 'Known ticket survived %'], axis=1, inplace=True)


def prepare_family_ticket_frequencies_not_leaked(data, is_train, train):

    # add temporary last name who it's going to be decided by if it's the same family
    data['Last name'] = data['Name'].apply(extract_lastname)
    train_last_names = train['Name'].apply(extract_lastname)

    data['Known family survived %'] = np.NaN
    data['Known ticket survived %'] = np.NaN
    data['Known family/ticket survived %'] = np.NaN
    data['Family/ticket survival known'] = 1

    # go over all test passengers, and fill in the survival information
    for i in data.index:
        did_survive = 1 if (is_train == 1) and (train.loc[i, 'Survived'] == 1) else 0
        last_name = data.loc[i, 'Last name']
        ticket = data.loc[i, 'Ticket']

        # if have other passengers in training set of same family whose survival information is known, copy average here
        if train[train_last_names == last_name]['Survived'].count() > is_train:
            data.loc[i, 'Known family survived %'] = \
                (train[train_last_names == last_name]['Survived'].sum() - did_survive) / \
                (train[train_last_names == last_name]['Survived'].count() - is_train)

        # if have other passengers in training set of same family whose survival information is known, copy average here
        # add information for training only of how many of known survived in the same ticket
        if train[train['Ticket'] == ticket]['Survived'].count() > is_train:
            data.loc[i, 'Known ticket survived %'] = \
                (train[train['Ticket'] == ticket]['Survived'].sum() - did_survive) / \
                (train[train['Ticket'] == ticket]['Survived'].count() - is_train)

        # For final value of
        if np.isnan(data.loc[i, 'Known family survived %']) == False:
            if np.isnan(data.loc[i, 'Known ticket survived %']) == False:
                # both family and ticket survival rates known, take average
                data.loc[i, 'Known family/ticket survived %'] = \
                    (data.loc[i, 'Known family survived %'] + data.loc[i, 'Known ticket survived %']) / 2
            else:
                # only family survival known, take it
                data.loc[i, 'Known family/ticket survived %'] = data.loc[i, 'Known family survived %']
        elif np.isnan(data.loc[i, 'Known ticket survived %']) == False:
            # only ticket is known - take value from ticket
            data.loc[i, 'Known family/ticket survived %'] = data.loc[i, 'Known ticket survived %']
        else:
            # none known, set mean survival value
            data.loc[i, 'Known family/ticket survived %'] = train['Survived'].mean()
            data.loc[i, 'Family/ticket survival known'] = 0

    print(f'Debug: survival rates: is_train: {is_train}\n'
          f'{data["Known family/ticket survived %"].value_counts(dropna=False)}')

    # drop temporary columns used
    data.drop(['Last name', 'Known family survived %', 'Known ticket survived %'], axis=1, inplace=True)


def manual_age_bin(age):
    if age <= 4:
        return '-4'
    elif age <= 11:
        return '4-11'
    elif age <= 24:
        return '11-24'
    elif age <= 32:
        return '24-32'
    elif age <= 42:
        return '32-42'
    else:
        return '42+'


# TODO - should leave?
def prepare_features_scale_train_test_together(train, x_test, output_folder):
    num_train_samples = train.shape[0]

    print(f'Debug: RANDOM_STATE:{RANDOM_STATE_FIRST}, num_train_samples:{num_train_samples}')
    print(f'Debug: Features before adding / dropping: {x_test.columns.values}')

    features_to_drop_after_use = []
    features_to_add_dummies = []
    both = combine_train_x_test(train, x_test)

    # 1 ---> Adding title, see details in Advanced feature engineering.ipynb
    both['Title'] = both['Name'].apply(get_title).replace(title_map)
    features_to_add_dummies.append('Title')
    features_to_drop_after_use.append('Name')

    # 2 ---> Create a new feature of number 'Family size' of relatives regardless of who they are
    #   Group SibSp, Parch, Family size based on different survival rates
    both['Family size'] = 1 + both['SibSp'] + both['Parch']
    if 'SibSp' not in options['major_columns_to_drop']:
        both['SibSpBin'] = both['SibSp'].replace({0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                                                  5: '5+', 8: '5+'})
        features_to_add_dummies.append('SibSpBin')
        features_to_drop_after_use.append('SibSp')
    if 'Parch' not in options['major_columns_to_drop']:
        both['ParchBin'] = both['Parch'].replace({0: '0',
                                                  1: '1',
                                                  2: '2',
                                                  3: '3',
                                                  4: '4+', 5: '4+', 6: '4+', 9: '4+'})
        features_to_add_dummies.append('ParchBin')
        features_to_drop_after_use.append('Parch')

    if 'Family size' not in options['major_columns_to_drop']:
        both['Family size'] = both['Family size'].replace({1: '1',
                                                           2: '23', 3: '23',
                                                           4: '4',
                                                           5: '567', 6: '567', 7: '567',
                                                           8: '8+', 11: '8+'})
        features_to_add_dummies.append('Family size')

    # 3. ----> Prepare Deck features based on first letter of Cabin
    both['Cabin'] = both['Cabin'].fillna('unknown')
    both['Deck'] = both['Cabin'].apply(lambda cab: cab[0] if (cab != 'unknown') else cab)
    both['DeckBin'] = both['Deck'].replace({'unknown': 'unknown_T', 'T': 'unknown_T',
                                            'B': 'B',
                                            'D': 'DE', 'E': 'DE',
                                            'C': 'CF', 'F': 'CF',
                                            'A': 'AG', 'G': 'AG'})
    features_to_drop_after_use.append('Cabin')
    features_to_drop_after_use.append('Deck')
    features_to_add_dummies.append('DeckBin')

    # 4 ---> Add Pclass category
    features_to_add_dummies.append('Pclass')

    # 5 ---> Add Sex
    if 'Sex' not in options['major_columns_to_drop']:
        both['Sex'] = both['Sex'].map({'male': 1, 'female': 0})

    # 6 ---> Add Embarked, fill the 2 missing values with the most common S
    both['Embarked'] = both['Embarked'].fillna('S')  # needed anyways for imputing age
    if 'Embarked' not in options['major_columns_to_drop']:
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
    #   Currently decided based on feature importance to only leave Fare 13.5
    both['Fare 13.5+'] = both['Fare per person'].apply(lambda fare: 1 if fare > 13.5 else 0)
    both['Fare log'] = both['Fare per person'].replace({0: 0.0001})  # to avoid doing log on 0 which is invalid
    both['Fare log'] = np.log(both['Fare log'])
    features_to_drop_after_use.append('Fare')
    features_to_drop_after_use.append('Fare per person')

    # 9 --> Add frequencies of survival per family (based on last name) and ticket
    prepare_family_ticket_frequencies_leaked(both, train['Survived'])

    # 10 --> Age - fill in missing values, bin
    impute_age_regression_train_test(both)
    both['Age Bin'] = both['Age'].apply(manual_age_bin)
    print(f"Debug: Age value_counts:\n{both['Age Bin'].value_counts().sort_index()}")
    features_to_add_dummies.append('Age Bin')

    print(f'Debug: Features before dropping not used at all, '
          f'shape {both.shape}: {both.columns.values}')

    both.drop(features_to_drop_after_use, axis=1, inplace=True)

    print(f'Debug: Features after dropping not used at all, before major dropping, '
          f'shape {both.shape}: {both.columns.values}')

    both.drop(options['major_columns_to_drop'], axis=1, inplace=True)

    print(f'Debug: Features after dropping major, before dummies, shape {both.shape}: {both.columns.values}')

    both = pd.get_dummies(both, columns=features_to_add_dummies)

    print(f'Debug: Features after dummies before dropping minor, shape {both.shape}: {both.columns.values}')

    both.drop(options['minor_columns_to_drop'], axis=1, inplace=True)

    print(f'Debug: Features after dummies after dropping minor, shape {both.shape}: {both.columns.values}')

    print(f'Debug: both.info:\n{both.info()}')
    print(f'Debug: Value counts of all values:')
    for feat in both.columns.values:
        print(f'--------------- {feat}:')
        print(both[feat].value_counts())

    # TODO - should it be here?
    both.corr().to_csv(output_folder + 'feature_correlations.csv')

    new_x_train, new_x_test = split_into_x_train_x_test(both)

    scaler = StandardScaler()
    scaler.fit(new_x_train)
    x_train_scaled = scaler.transform(new_x_train)
    x_test_scaled = scaler.transform(new_x_test)

    return x_train_scaled, x_test_scaled, new_x_train.columns


def prepare_features_scale_based_on_train_only(data, is_train, train, scaler):

    print(f'Debug: Preparing features of data of shape {data.shape}, is_train: {is_train}')
    print(f'Debug: Features before adding / dropping: {data.columns.values}')

    # not to override the original data, since feature preparation will be done numerous times
    data = data.copy()

    features_to_drop_after_use = []
    features_to_add_dummies = []

    # 1 ---> Adding title, see details in Advanced feature engineering.ipynb
    # TODO would you do the same map looking at only part of the data?
    data['Title'] = data['Name'].apply(get_title).replace(
        {'Lady': 'Mrs', 'Mme': 'Mrs', 'Dona': 'Mrs', 'the Countess': 'Mrs',
         'Ms': 'Miss', 'Mlle': 'Miss',
         'Sir': 'Mr', 'Major': 'Mr', 'Capt': 'Mr', 'Jonkheer': 'Mr', 'Don': 'Mr', 'Col': 'Mr', 'Rev': 'Mr', 'Dr': 'Mr'})
    features_to_add_dummies.append('Title')
    features_to_drop_after_use.append('Name')

    # 2 ---> Create a new feature of number 'Family size' of relatives regardless of who they are
    #   Group SibSp, Parch, Family size based on different survival rates
    data['Family size'] = 1 + data['SibSp'] + data['Parch']
    if 'SibSp' not in options['major_columns_to_drop']:
        # TODO would you do the same map looking at only part of the data?
        data['SibSpBin'] = data['SibSp'].replace({0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                                                  5: '5+', 8: '5+'})
        features_to_add_dummies.append('SibSpBin')
        features_to_drop_after_use.append('SibSp')
    if 'Parch' not in options['major_columns_to_drop']:
        # TODO would you do the same map looking at only part of the data?
        data['ParchBin'] = data['Parch'].replace({0: '0',
                                                  1: '1',
                                                  2: '2',
                                                  3: '3',
                                                  4: '4+', 5: '4+', 6: '4+', 9: '4+'})
        features_to_add_dummies.append('ParchBin')
        features_to_drop_after_use.append('Parch')

    if 'Family size' not in options['major_columns_to_drop']:
        # TODO would you do the same map looking at only part of the data?
        data['Family size'] = data['Family size'].replace({1: '1',
                                                           2: '23', 3: '23',
                                                           4: '4',
                                                           5: '567', 6: '567', 7: '567',
                                                           8: '8+', 11: '8+'})
        features_to_add_dummies.append('Family size')

    # 3. ----> Prepare Deck features based on first letter of Cabin
    data['Cabin'] = data['Cabin'].fillna('unknown')
    data['Deck'] = data['Cabin'].apply(lambda cab: cab[0] if (cab != 'unknown') else cab)
    # TODO would you do the same map looking at only part of the data?
    data['DeckBin'] = data['Deck'].replace({'unknown': 'unknown_T', 'T': 'unknown_T',
                                            'B': 'B',
                                            'D': 'DE', 'E': 'DE',
                                            'C': 'CF', 'F': 'CF',
                                            'A': 'AG', 'G': 'AG'})
    features_to_drop_after_use.append('Cabin')
    features_to_drop_after_use.append('Deck')
    features_to_add_dummies.append('DeckBin')

    # 4 ---> Add Pclass category
    features_to_add_dummies.append('Pclass')

    # 5 ---> Add Sex
    if 'Sex' not in options['major_columns_to_drop']:
        data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

    # 6 ---> Add Embarked, fill the 2 missing values with the most common S
    data['Embarked'] = data['Embarked'].fillna('S')  # needed anyways for imputing age
    if 'Embarked' not in options['major_columns_to_drop']:
        features_to_add_dummies.append('Embarked')

    # 7 --> Add new feature of Ticked Frequency - how many times this ticket appeared,
    #   kind of size of family but of ticket
    # TODO:
    # A leaked feature, consider re-adding later as an option
    # both['Ticket_Frequency'] = both.groupby('Ticket')['Ticket'].transform('count')
    features_to_drop_after_use.append('Ticket')

    # 8 --> Fare. Add new category of "Fare per person" since fares are currently per ticket, and Set missing values
    #   Replace with bins, and get rid of regular Fare
    # TODO - was based on 'Ticket_Frequency' and not 'Family size', if Ticket_Frequency returned, consider adding back
    data['Fare per person'] = data['Fare'] / data['Family size']
    # In the same class, Fare per person has a tight distribution, so just take median
    train_class3 = train[train['Pclass'] == 3]
    imputed_fare = (train_class3['Fare'] / (train_class3['SibSp'] + train_class3['Parch'] + 1)).median()
    data['Fare per person'] = data['Fare per person'].fillna(imputed_fare)
    # Since the missing Fare is only of a person with Family size 1, take median of class 3 of Fare per person
    data['Fare'] = data['Fare'].fillna(imputed_fare)
    # Add categorical category of manual bins of fares (see Advanced feature engineering.ipynb notebook)
    #   Currently decided based on feature importance to only leave Fare 13.5
    # TODO would you do the same map looking at only part of the data?
    data['Fare 13.5+'] = data['Fare per person'].apply(lambda fare: 1 if fare > 13.5 else 0)
    data['Fare log'] = data['Fare per person'].replace({0: 0.0001})  # to avoid doing log on 0 which is invalid
    data['Fare log'] = np.log(data['Fare log'])
    features_to_drop_after_use.append('Fare')
    features_to_drop_after_use.append('Fare per person')

    # 9 --> Add frequencies of survival per family (based on last name) and ticket
    prepare_family_ticket_frequencies_not_leaked(data, is_train, train)

    # 10 --> Age - fill in missing values, bin
    impute_age_regression_based_on_train(data, train)
    data['Age Bin'] = data['Age'].apply(manual_age_bin)
    print(f"Debug: Age value_counts:\n{data['Age Bin'].value_counts().sort_index()}")
    features_to_add_dummies.append('Age Bin')

    print(f'Debug: Features before dropping not used at all, '
          f'shape {data.shape}: {data.columns.values}')

    data.drop(features_to_drop_after_use, axis=1, inplace=True)

    print(f'Debug: Features after dropping not used at all, before major dropping, '
          f'shape {data.shape}: {data.columns.values}')

    data.drop(options['major_columns_to_drop'], axis=1, inplace=True)

    print(f'Debug: Features after dropping major, before dummies, shape {data.shape}: {data.columns.values}')

    data = pd.get_dummies(data, columns=features_to_add_dummies)

    print(f'Debug: Features after dummies before dropping minor, shape {data.shape}: {data.columns.values}')

    data.drop(options['minor_columns_to_drop'], axis=1, inplace=True)

    print(f'Debug: Features after dummies after dropping minor, shape {data.shape}: {data.columns.values}')

    print(f'Debug: both.info:\n{data.info()}')
    print(f'Debug: Value counts of all values:')
    for feat in data.columns.values:
        print(f'--------------- {feat}:')
        print(data[feat].value_counts())

    # TODO - should it be here?
    # TODO used to print to file. Can still print to file if correlations are different multiple times due to different data?
    #   File will be overridden? Doesn't make sense to save per permutation.  Perhaps only to print
    # TODO make sure all data is printed properly and seen on the display
    print(data.corr())

    if is_train:
        scaler = StandardScaler()
        scaler.fit(data)

    data_scaled = scaler.transform(data)

    return data_scaled, scaler, data.columns


def output_all_preds(preds, x_test, output_folder):
    preds_dir = output_folder + 'preds/'
    os.mkdir(preds_dir)
    for pred_name in preds:
        pred_df = pd.DataFrame(preds[pred_name])
        pred_df.set_index(x_test.index, inplace=True)
        pred_df.columns = ['Survived']
        pred_df.to_csv(f'{preds_dir}preds_{pred_name}.csv')


def dif_detailed_with_folds(name_str, type_class, classifier, x_train, y_train, x_test, train, results, preds,
                            train_probas, test_probas, start_time):

    for cv_fold in options['cv_folds']:
        fit_detailed(name_str, type_class, classifier, x_train, y_train, x_test, train,
                     results, preds, train_probas, test_probas, start_time, cv_fold)


def get_cross_val_score_various_rands(classifier, x_train, y_train, cv_folds, num_rands):
    accuracies_with_rand = []

    for rand_loop in range(num_rands):
        rand = rand_loop + RANDOM_STATE_FIRST
        classifier.set_params(random_state=rand_loop + RANDOM_STATE_FIRST)

        fold = KFold(cv_folds, True, random_state=rand)

        for train_indicies, test_indicies in fold.split(x_train, y_train):
            classifier.fit(x_train[train_indicies], y_train.reset_index(drop=True)[train_indicies])
            accuracy = classifier.score(x_train[test_indicies], y_train.reset_index(drop=True)[test_indicies])
            accuracies_with_rand.append(accuracy)

    return np.mean(accuracies_with_rand), np.std(accuracies_with_rand)


def fit_detailed(name_str, type_class, classifier, x_train, y_train, x_test, train, results, preds,
                 train_probas, test_probas, start_time, cv_folds):

    if options['features_based_on_train_only']:
        x_train_scaled, scaler, columns = prepare_features_scale_based_on_train_only(x_train, True, train, None)
        x_test_scaled, _, _ = prepare_features_scale_based_on_train_only(x_test, False, train, scaler)
    else:
        x_train_scaled = x_train
        x_test_scaled = x_test

    cross_acc_score, cross_acc_std = get_cross_val_score_various_rands(
        classifier, x_train_scaled, y_train, cv_folds, options['num_rands'])

    classifier.fit(x_train_scaled, y_train)
    preds[name_str] = classifier.predict(x_test_scaled)
    train_acc_score = classifier.score(x_train_scaled, y_train)

    train_preds = classifier.predict(x_train_scaled)
    train_roc_auc_score = round(roc_auc_score(y_train, classifier.predict_proba(x_train_scaled)[:, 1]), 2)
    train_f1_score_not_survived = round(f1_score(y_train, train_preds, average="micro", labels=[0]), 2)
    train_f1_score_survived = round(f1_score(y_train, train_preds, average="micro", labels=[1]), 2)
    train_precision_score_not_survived = round(precision_score(y_train, train_preds, average="micro", labels=[0]), 2)
    train_precision_score_survived = round(precision_score(y_train, train_preds, average="micro", labels=[1]), 2)
    train_recall_score_not_survived = round(recall_score(y_train, train_preds, average="micro", labels=[0]), 2)
    train_recall_score_survived = round(recall_score(y_train, train_preds, average="micro", labels=[1]), 2)

    try:
        train_probas[name_str] = classifier.predict_proba(x_train_scaled)[:, 0]
        test_probas[name_str] = classifier.predict_proba(x_test_scaled)[:, 0]
    except AttributeError:
        # For Hard voting probabilities where predict_proba is not supported
        train_probas[name_str] = np.mean(x_train_scaled, axis=1)
        test_probas[name_str] = np.mean(x_test_scaled, axis=1)

    cross_acc_min_3_std = cross_acc_score - cross_acc_std * 3

    results.append({'Name': name_str,
                    'CV folds': cv_folds,
                    'Train acc': round(train_acc_score * 100, 1),
                    'Cross acc': round(cross_acc_score * 100, 1),
                    'Cross acc STD': round(cross_acc_std * 100, 1),
                    'Cross acc - 3*STD': round(cross_acc_min_3_std * 100, 1),
                    'Train - Cross acc-STD*3': round((train_acc_score - cross_acc_min_3_std) * 100, 1),
                    'Time sec': round(time.time() - start_time),
                    'Train auc': train_roc_auc_score,
                    'Train f1 died': train_f1_score_not_survived,
                    'Train f1 survived': train_f1_score_survived,
                    'Train precision died': train_precision_score_not_survived,
                    'Train precision survived': train_precision_score_survived,
                    'Train recall died': train_recall_score_not_survived,
                    'Train recall survived': train_recall_score_survived,
                    'Classifier options': classifier.get_params()})
    print(f'Debug: Stats {type_class}: {results[-1]}')

    if options['features_based_on_train_only']:
        print_feature_importances(name_str, classifier, columns)


def fit_grid_classifier(name_str, x_train, y_train, x_test, train, single_classifier, grid_params, results, preds,
                        train_probas, test_probas):
    start_time = time.time()

    grid = GridSearchCV(single_classifier, grid_params, verbose=1, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)
    classifier = grid.best_estimator_
    print(f'Debug: {name_str} best classifier:\n{classifier}')

    dif_detailed_with_folds(name_str, 'Grid', classifier, x_train, y_train, x_test, train, results, preds,
                      train_probas, test_probas, start_time)

    return classifier


def fit_single_classifier(name_str, x_train, y_train, x_test, train, classifier, results, preds, train_probas, test_probas):
    start_time = time.time()

    dif_detailed_with_folds(name_str, 'Single', classifier, x_train, y_train, x_test, train, results, preds,
                      train_probas, test_probas, start_time)

    return classifier


def fit_predict_voting(classifiers, name_str, voting_type, x_train, y_train, x_test, train, results, preds,
                       train_probas, test_probas):
    start_time = time.time()

    classifier = VotingClassifier(estimators=classifiers, voting=voting_type, n_jobs=-1)

    dif_detailed_with_folds(name_str, 'Voting', classifier, x_train, y_train, x_test, train, results, preds,
                      train_probas, test_probas, start_time)

    return classifier


def print_feature_importances(cl, classifier, columns):
    if 'Log' in cl:
        importances = pd.DataFrame({'Importance': classifier.coef_[0]}, index=columns). \
            reset_index().sort_values(by='Importance', ascending=False)
        print(f'Debug: "{cl}" feature importances:\n{pd.DataFrame(importances)}')
        importances['Importance'] = importances['Importance'].abs()
        print(f'Debug: "{cl}" feature importances (abs):\n'
              f'{pd.DataFrame(importances).sort_values(by="Importance", ascending=False).reset_index()}')
    elif 'RF' in cl:
        importances = pd.DataFrame({'Importance': classifier.feature_importances_}, index=columns).\
            reset_index().sort_values(by='Importance', ascending=False).reset_index()
        print(f'Debug: "{cl}" feature importances:\n{importances}')
    elif 'XGB' in cl:
        importance = pd.DataFrame(classifier.get_booster().get_score(importance_type="gain"),
                                  index=["Importance"]).transpose()
        print(f'Debug: "{cl}" feature importances:\n'
              f'{importance.sort_values(by="Importance", ascending=False).reset_index()}')


def write_to_file_input_options(output_folder):
    w = csv.writer(open(output_folder + 'input_options.csv', 'w', newline=''))
    for key, val in options.items():
        if key not in options['input_options_not_to_output']:
            w.writerow([key, val])


def main():
    start_time_total = time.time()

    output_folder = 'output/_' + time.strftime("%Y_%m_%d_%H_%M_%S") + '/'
    os.mkdir(output_folder)

    write_to_file_input_options(output_folder)

    train, x_test = read_files()
    x_train = train.drop('Survived', axis=1)
    y_train = train['Survived']

    if not options['features_based_on_train_only']:
        x_train, x_test, columns = prepare_features_scale_train_test_together(train, x_test, output_folder)

    results = []
    preds = pd.DataFrame()
    train_probas = pd.DataFrame()
    test_probas = pd.DataFrame()

    single_classifiers = options['single_classifiers']
    grid_classifiers = options['grid_classifiers']
    grid_classifiers_not_for_ensembling = options['grid_classifiers_not_for_ensembling']

    classifiers_for_ensembling = []

    # Unused prediction probabilities - to prevent taking some classifiers into account for voting and ensemble
    unused_train_proba = pd.DataFrame()
    unused_test_proba = pd.DataFrame()

    for cl in single_classifiers:
        classifier = fit_single_classifier(cl,
                                           x_train,
                                           y_train,
                                           x_test,
                                           train,
                                           single_classifiers[cl]['clas'],
                                           results, preds, unused_train_proba, unused_test_proba)
        # Currently, only using Grid classifiers for voting
        '''
        if cl not in grid_classifiers_not_for_ensembling:
            classifiers_for_ensembling.append((cl, classifier))
        '''
        # print feature importances for classifiers where it's easy to get this information
        # TODO should keep?
        if not options['features_based_on_train_only']:
            print_feature_importances(cl, classifier, columns)

    for cl in grid_classifiers:
        classifier = fit_grid_classifier(
            cl,
            x_train,
            y_train,
            x_test,
            train,
            grid_classifiers[cl]['clas'],
            grid_classifiers[cl]['grid_params'],
            results,
            preds,
            train_probas if cl not in grid_classifiers_not_for_ensembling else unused_train_proba,
            test_probas if cl not in grid_classifiers_not_for_ensembling else unused_test_proba)
        if cl not in grid_classifiers_not_for_ensembling:
            classifiers_for_ensembling.append((cl, classifier))

    # Ensembling from previous classifiers
    # Voting based on part of the Grid results - see grid_classifiers_not_for_ensembling

    if classifiers_for_ensembling:
        fit_predict_voting(classifiers_for_ensembling, 'Voting soft - part of grid', 'soft',
                           x_train, y_train, x_test, train,
                           results, preds, unused_train_proba, unused_test_proba)
        fit_predict_voting(classifiers_for_ensembling, 'Voting hard - part of grid', 'hard',
                           x_train, y_train, x_test, train,
                           results, preds, unused_train_proba, unused_test_proba)

    # Ensembling based on probabilities of previous classifiers
    # Based on part of the Grid results - see grid_classifiers_not_for_ensembling

    print(f'Debug: shape of train_probas: {train_probas.shape}, test_probas: {test_probas.shape}')
    print(f'Debug: head of train_probas:\n{train_probas.head()}')

    if train_probas.shape[0] > 0:
        # TODO - signature was changed to include train - fix
        fit_grid_classifier('Ensemble RF - part of grid', train_probas, y_train, test_probas,
                            RandomForestClassifier(n_estimators=1000, random_state=RANDOM_STATE_FIRST, n_jobs=-1),
                            [{'max_depth': range(3, 10)}],
                            results, preds, unused_train_proba, unused_test_proba)

        # TODO - signature was changed to include train - fix
        fit_grid_classifier('Ensemble Log - part of grid', train_probas, y_train, test_probas,
                            LogisticRegression(solver='liblinear', random_state=RANDOM_STATE_FIRST, n_jobs=-1),
                            [{'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']}],
                            results, preds, unused_train_proba, unused_test_proba)

    preds.corr().to_csv(output_folder + 'classifiers_correlations.csv')

    if options['output_preds']:
        output_all_preds(preds, x_test, output_folder)

    pd.DataFrame(results).to_csv(output_folder + 'results.csv')

    print(f'Debug: Time took: {time.time() - start_time_total} seconds = '
          f'{round((time.time() - start_time_total) / 60)} minutes ')


'''
TODO:
Beginning:
- Do all feature preparation on each fold separately - both train and test, and each fold of the train.  This will prevent leakage, but it will actually probably lower the score
- k-Fold actual training (in addition to Bagging? Instead?) How to actually combine results?
- Consider Bagging and not just cross-validation at one of the lower levels
    Use Out of Bag accuracy when doing Bagging
- Do different views of the features (what's included / not included / in what format)
A bit later:
- Take code and ideas from https://machinelearningmastery.com/spot-check-machine-learning-algorithms-in-python/
- First do Grid, then cross-validation
- Automate bottom line report and choosing of the model
- Do feature selection with RFECV per algorithm
- Try stratified fold
Make sure didn't break after this step: 
- Make sure doesn't break: Shuffle with random_state - some algorithms are effected by the order, random state for cross-validation
- Make sure actually using different cross-validation sizes
- Make sure using average of a few random sizes 
Middle:
- Add extra trees algorithm, AdaBoost, Bernoulli NB (perhaps instead / in addition to Gaussasian NB), others from his list of best / all
    From his list of algorithms for classification: Random Forest, XGBoost, SVM, (Backpropogation - what specifically is it?), Decision Trees (CART and C4.5/C5.0), Naive Bayes, Logistic Regression and Linear Discriminant Analysis, k-Nearest Neighbors and Learning Vector Quantization (what is it?)
- Give a chance to each one of the classifiers
- XGBoost - do much more parameter optimizations
- Read in my summaries what other steps need to try 
End:
- Voting only on models I know work best
- Consider using statistical tests to decide with algorithm is better: parametric / non parametric, P-value

'''

options = {
    'output_preds': False,
    # TODO - need to somehow print options of the grid in a useful way
    'input_options_not_to_output': ['single_classifiers', 'grid_classifiers'],
    'cv_folds': [2, 3, 5, 10],  # options of number of folds for Cross validation
    'num_rands': 15,  # number of times to run the same thing with various random numbers
    'features_based_on_train_only': True,
    # main columns to drop
    'major_columns_to_drop': [
        'Sex',  # Since titles are important, need to remove Sex
        'Family/ticket survival known',  # low in all models
        'SibSp',  # very low in all models, perhaps because of Family size / Ticket_Frequency
        'Parch',  # very low in all models, perhaps because of Family size / Ticket_Frequency
        'Embarked',
        'Fare log',  # seems doesn't make the model stable, causes overfitting, doesn't add much to the model
        'Family size',  # high correlation with Ticket_Frequency that wins for all models
        'Age'  # but adding Age bins

    ],
    # specific binned features to drop, like a specific bin in the main feature
    'minor_columns_to_drop': [
        # -- Age - not extemely important, most models Age_-4 is important (15), XGB gives more age importance (6,8)
        # 'Age Bin_-4',
        'Age Bin_4-11',  # low in all classifiers
        # 'Age Bin_11-24',
        # 'Age Bin_24-32',
        # 'Age Bin_32-42',
        # 'Age Bin_42+'

        # --- Fare bin
        # 'Fare bin_13.5+'  # important, places 2-10

        # -- Deck - some important, some not. what's left is important, unknown_T and DE
        'DeckBin_AG',
        'DeckBin_B',
        'DeckBin_CF',
        # 'DeckBin_DE'  # important, perhaps because of mixed deck and more change for non 1st class to survive
        # 'DeckBin_unknown_T'  # important, especially low survival

        # -- Title
        'Title_Master'  # WAS NOT NEEDED FOR Random Forest (perhaps because of age connection)

        # -- Pclass - important in most classifiers
        # -- Ticket_Frequency - place 7,10,14, leaving
        # -- Known family/ticket survived % - places 2,4 - one of the most important
    ],
    # classifiers that we don't use for Grid Search
    'single_classifiers': {
        'Log': {'clas': LogisticRegression(solver='liblinear', random_state=RANDOM_STATE_FIRST, n_jobs=-1)},
#        'KNN 14': {'clas': KNeighborsClassifier(n_neighbors=14, n_jobs=-1)},
#        'SVM rbf': {'clas': SVC(gamma='auto', kernel='rbf', probability=True, random_state=RANDOM_STATE)},
#        'SVM poly': {'clas': SVC(gamma='auto', kernel='poly', probability=True, random_state=RANDOM_STATE)},
#        'NB': {'clas': GaussianNB()},  # consistently gives worse results
#        'RF 10': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)},
#        'RF 9': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=9, random_state=RANDOM_STATE, n_jobs=-1)},
#        'RF 8': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)},
#        'RF 7': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=RANDOM_STATE, n_jobs=-1)},
#        'RF 6': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)},
#        'XGB': {'clas': xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000,
 #                                         random_state=RANDOM_STATE, n_jobs=-1)}
    },
    # Classifiers we use with Grid search
    'grid_classifiers': {
 #       'Grid Log': {'clas': LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, n_jobs=-1),
 #                    'grid_params': [{'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']}]},
 #       'Grid KNN': {'clas': KNeighborsClassifier(n_neighbors=14, n_jobs=-1),
 #                    'grid_params': [{'n_neighbors': range(3, 25)}]},
 #       'Grid SVM': {'clas': SVC(gamma='auto', kernel='rbf', probability=True, random_state=RANDOM_STATE),
 #                    'grid_params':
 #                        [{
 #                           'kernel': ['rbf', 'poly', 'sigmoid'],
 #                           'C': [0.3, 0.5, 1.0, 1.5, 2.0],
 #                           'gamma': [0.3, 0.2, 0.1, 0.05, 0.01, 'auto_deprecated', 'scale']
 #                        }],
 #                    },
 #       'Grid RF': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=RANDOM_STATE, n_jobs=-1),
 #                   'grid_params': [{'max_depth': range(3, 10)}]},
 #       'Grid XGB': {'clas': xgb.XGBClassifier(objective='binary:logistic',
 #                                              n_estimators=1000,
 #                                              random_state=RANDOM_STATE,
 #                                              n_jobs=-1),
 #                    'grid_params':
 #                        [{
 #                            'max_depth': range(1, 8, 1)  # default 3 - higher depth - less bias, more variance
 #                            # 'n_estimators': range(60, 260, 40), # default 100
 #                            # 'learning_rate': [0.3, 0.2, 0.1, 0.01],  # , 0.001, 0.0001
 #                            # 'min_child_weight': [0.5, 1, 2],  # default 1 - higher number, less overfitting, when to stop splitting the child given sum of weights
 #                            # 'subsample': [i / 10.0 for i in range(6, 11)], # default 1, smaller values prevent overfitting
 #                            # 'colsample_bytree': [i / 10.0 for i in range(6, 11)] # default 1, fraction of features selected for each tree
 #                            # 'gamma': [i / 10.0 for i in range(3)]  # default 0 - for what gain in metric to continue splitting
 #                        }]
 #                    }
    },
    'grid_classifiers_not_for_ensembling': ['Grid SVM', 'Grid XGB']
}

# TODO - make nicer
title_map = {'Lady': 'Mrs', 'Mme': 'Mrs', 'Dona': 'Mrs', 'the Countess': 'Mrs',
         'Ms': 'Miss', 'Mlle': 'Miss',
         'Sir': 'Mr', 'Major': 'Mr', 'Capt': 'Mr', 'Jonkheer': 'Mr', 'Don': 'Mr', 'Col': 'Mr', 'Rev': 'Mr', 'Dr': 'Mr'}

main()
