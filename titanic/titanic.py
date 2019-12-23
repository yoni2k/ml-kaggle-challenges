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

RANDOM_STATE_FIRST = 50


def read_files():
    train = pd.read_csv('input/train.csv', index_col='PassengerId')
    x_test = pd.read_csv('input/test.csv', index_col='PassengerId')
    return train, x_test


def get_title(full_name):
    return full_name.split(',')[1].split('.')[0].strip()


def extract_lastname(full_name):
    return full_name.split(',')[0]


def impute_age_regression(full_x_train, full_x_test):
    # Conclusions:
    # - Sex doesn't seem to help (because there is title)
    # - Didn't try, harder to add, and probably won't help much: 'Fare bin', 'Deck' has size issues
    # TODO - used to be also based on leaked feature of Ticket_Frequency
    features_base_age_on = ['Pclass IMPUTE', 'Title IMPUTE', 'Parch IMPUTE', 'SibSp IMPUTE', 'Embarked IMPUTE']

    age_predict_train = full_x_train.loc[full_x_train['Age'].isnull(), features_base_age_on]
    age_predict_test = full_x_test.loc[full_x_test['Age'].isnull(), features_base_age_on]
    age_x_train = full_x_train.loc[full_x_train['Age'].isnull() == False, features_base_age_on]
    age_y_train = full_x_train.loc[full_x_train['Age'].isnull() == False, 'Age']

    for cat in ['Title IMPUTE', 'Embarked IMPUTE']:
        labelenc = LabelEncoder().fit(age_x_train[cat])
        age_x_train[cat] = labelenc.transform(age_x_train[cat])
        age_predict_train[cat] = labelenc.transform(age_predict_train[cat])
        age_predict_test[cat] = labelenc.transform(age_predict_test[cat])

    age_model = RandomForestRegressor(n_estimators=100)
    age_model.fit(age_x_train, age_y_train)
    print(f"Debug: Age prediction feature importance, features: {features_base_age_on}, "
          f"importance:\n{age_model.feature_importances_}")

    full_x_train.loc[full_x_train['Age'].isnull(), 'Age'] = age_model.predict(age_predict_train)
    full_x_test.loc[full_x_test['Age'].isnull(), 'Age'] = age_model.predict(age_predict_test)

    print(f'Debug: Age prediction score: {round(age_model.score(age_x_train, age_y_train) * 100, 1)}')


def prepare_family_ticket_frequencies_actual(data, is_train, train, last_names_survival, tickets_survival):
    data['Known family survived %'] = np.NaN
    data['Known ticket survived %'] = np.NaN
    data['Known family/ticket survived %'] = np.NaN
    data['Family/ticket survival known'] = 1

    mean_train_survive = train['Survived'].mean()

    # go over all test passengers, and fill in the survival information
    for i in data.index:
        did_survive = 1 if (is_train == 1) and (train.loc[i, 'Survived'] == 1) else 0
        last_name = data.loc[i, 'Last name IMPUTE']
        ticket = data.loc[i, 'Ticket IMPUTE']

        # if have other passengers in training set of same family whose survival information is known, copy average here

        if last_name in last_names_survival:
            last_name_count, last_name_sum = last_names_survival[last_name]
            if last_name_count > is_train:
                data.loc[i, 'Known family survived %'] = (last_name_sum - did_survive) / (last_name_count - is_train)

        # if have other passengers in training set of same family whose survival information is known, copy average here
        # add information for training only of how many of known survived in the same ticket
        if ticket in tickets_survival:
            ticket_count, ticket_sum = tickets_survival[ticket]
            if ticket_count > is_train:
                data.loc[i, 'Known ticket survived %'] = (ticket_sum - did_survive) / (ticket_count - is_train)

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
            data.loc[i, 'Known family/ticket survived %'] = mean_train_survive
            data.loc[i, 'Family/ticket survival known'] = 0

    print(f'Debug: survival rates: is_train: {is_train}\n'
          f'{data["Known family/ticket survived %"].value_counts(dropna=False)}')

    # drop temporary columns used
    data.drop(['Known family survived %', 'Known ticket survived %'], axis=1, inplace=True)


def prepare_family_ticket_frequencies(x_train, x_test, train):

    last_names_survival = {}

    for last_name in (list(x_train['Last name IMPUTE'].unique()) + list(x_test['Last name IMPUTE'].unique())):
        last_name_survived = train[train['Last name IMPUTE'] == last_name]['Survived']
        if last_name_survived.shape[0] > 0:
            last_names_survival[last_name] = (last_name_survived.count(), last_name_survived.sum())

    tickets_survival = {}

    for ticket in (list(x_train['Ticket IMPUTE'].unique()) + list(x_test['Ticket IMPUTE'].unique())):
        ticket_survived = train[train['Ticket IMPUTE'] == ticket]['Survived']
        if ticket_survived.shape[0] > 0:
            tickets_survival[ticket] = (ticket_survived.count(), ticket_survived.sum())

    prepare_family_ticket_frequencies_actual(x_train, True, train, last_names_survival, tickets_survival)
    prepare_family_ticket_frequencies_actual(x_test, False, train, last_names_survival, tickets_survival)


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


def prepare_features_scale_once(data, train):
    print(f'Debug: ONCE Preparing features of data of shape {data.shape}')
    print(f'Debug: ONCE Features before adding / dropping: {data.columns.values}')

    features_to_drop_after_use = []
    features_to_add_dummies = []

    # 1 ---> Adding title, see details in Advanced feature engineering.ipynb
    # TODO would you do the same map looking at only part of the data?
    data['Title'] = data['Name'].apply(get_title).replace(
        {'Lady': 'Mrs', 'Mme': 'Mrs', 'Dona': 'Mrs', 'the Countess': 'Mrs',
         'Ms': 'Miss', 'Mlle': 'Miss',
         'Sir': 'Mr', 'Major': 'Mr', 'Capt': 'Mr', 'Jonkheer': 'Mr', 'Don': 'Mr', 'Col': 'Mr', 'Rev': 'Mr', 'Dr': 'Mr'})
    features_to_add_dummies.append('Title')

    # 2 ---> Create a new feature of number 'Family size' of relatives regardless of who they are
    #   Group SibSp, Parch, Family size based on different survival rates
    if 'SibSp' not in options['major_columns_to_drop_once']:
        # TODO would you do the same map looking at only part of the data?
        data['SibSpBin'] = data['SibSp'].replace({0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                                                  5: '5+', 8: '5+'})
        features_to_add_dummies.append('SibSpBin')
        features_to_drop_after_use.append('SibSp')
    if 'Parch' not in options['major_columns_to_drop_once']:
        # TODO would you do the same map looking at only part of the data?
        data['ParchBin'] = data['Parch'].replace({0: '0',
                                                  1: '1',
                                                  2: '2',
                                                  3: '3',
                                                  4: '4+', 5: '4+', 6: '4+', 9: '4+'})
        features_to_add_dummies.append('ParchBin')
        features_to_drop_after_use.append('Parch')

    if 'Family size' not in options['major_columns_to_drop_once']:
        data['Family size'] = 1 + data['SibSp'] + data['Parch']
        # TODO would you do the same map looking at only part of the data?
        if options['feature_view']['Family size'] != 'Num':
            data['Family size bin'] = data['Family size'].replace({1: '1',
                                                                   2: '23', 3: '23',
                                                                   4: '4',
                                                                   5: '567', 6: '567', 7: '567',
                                                                   8: '8+', 11: '8+'})
            features_to_add_dummies.append('Family size bin')
        if options['feature_view']['Family size'] == 'Bin':
            features_to_drop_after_use.append('Family size')

    # 3. ----> Prepare Deck features based on first letter of Cabin
    data['Cabin'] = data['Cabin'].fillna('unknown')
    data['Deck'] = data['Cabin'].apply(lambda cab: cab[0] if (cab != 'unknown') else cab)
    # TODO would you do the same map looking at only part of the data?
    data['DeckBin'] = data['Deck'].replace({'unknown': 'unknown_T', 'T': 'unknown_T',
                                            'B': 'B',
                                            'D': 'DE', 'E': 'DE',
                                            'C': 'CF', 'F': 'CF',
                                            'A': 'AG', 'G': 'AG'})
    features_to_drop_after_use.append('Deck')
    features_to_add_dummies.append('DeckBin')

    # 4 ---> Add Pclass category
    features_to_add_dummies.append('Pclass')

    # 5 ---> Add Sex
    if 'Sex' not in options['major_columns_to_drop_once']:
        data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})

    # 6 ---> Add Embarked, fill the 2 missing values with the most common S
    data['Embarked'] = data['Embarked'].fillna('S')  # needed anyways for imputing age
    if 'Embarked' not in options['major_columns_to_drop_once']:
        features_to_add_dummies.append('Embarked')

    # 7 --> Add new feature of Ticked Frequency - how many times this ticket appeared,
    #   kind of size of family but of ticket
    # TODO:
    # A leaked feature, consider re-adding later as an option
    # both['Ticket_Frequency'] = both.groupby('Ticket')['Ticket'].transform('count')

    # 8 --> Fare. Add new category of "Fare per person" since fares are currently per ticket, and Set missing values
    #   Replace with bins, and get rid of regular Fare
    #   Doint it once and not per every fold, since there is only 1 value missing - will not change much
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
    if 'Num' not in options['feature_view']['Fare']:
        features_to_drop_after_use.append('Fare per person')
    if '13.5' not in options['feature_view']['Fare']:
        features_to_drop_after_use.append('Fare 13.5+')
    if 'Log' not in options['feature_view']['Fare']:
        features_to_drop_after_use.append('Fare log')

    # TODO - think if there is a nicer way to do this
    # Copy relevant categories once, will be used for imputing age and family frequency, then deleted.
    # This way the logic of whether to leave them / delete them / put dummies instead can be done without worrying
    #   that it will hurt the imputing of age / ticket frequencies
    data['Last name IMPUTE'] = data['Name'].apply(extract_lastname)
    data['Ticket IMPUTE'] = data['Ticket']
    data['Pclass IMPUTE'] = data['Pclass']
    data['Title IMPUTE'] = data['Title']
    data['Parch IMPUTE'] = data['Parch']
    data['SibSp IMPUTE'] = data['SibSp']
    data['Embarked IMPUTE'] = data['Embarked']

    print(f'Debug: ONCE Features before dropping not used at all, '
          f'shape {data.shape}: {data.columns.values}')

    data.drop(features_to_drop_after_use, axis=1, inplace=True)

    print(f'Debug: ONCE Features after dropping not used at all, before major dropping, '
          f'shape {data.shape}: {data.columns.values}')

    data.drop(options['major_columns_to_drop_once'], axis=1, inplace=True)

    print(f'Debug: ONCE Features after dropping major, before dummies, shape {data.shape}: {data.columns.values}')

    data = pd.get_dummies(data, columns=features_to_add_dummies)

    print(f'Debug: ONCE Features after dummies before dropping minor, shape {data.shape}: {data.columns.values}')

    data.drop(options['minor_columns_to_drop_once'], axis=1, inplace=True)

    print(f'Debug: ONCE Features after dummies after dropping minor, shape {data.shape}: {data.columns.values}')

    print(f'Debug: ONCE data.info:\n{data.info()}')
    print(f'Debug: ONCE Value counts of all values:')
    for feat in data.columns.values:
        print(f'--------------- {feat}:')
        print(data[feat].value_counts())

    return data


def prepare_features_scale_every_time(x_train, x_test, train):

    print(f'Debug: EVERY_TIME Preparing features of data of shape: x_train: {x_train.shape}, x_test: {x_test.shape}')
    print(f'Debug: EVERY_TIME Features before adding / dropping: {x_train.columns.values}')

    features_to_drop_after_use = []
    features_to_add_dummies = []

    # 9 --> Add frequencies of survival per family (based on last name) and ticket
    time_stamp = time.time()
    prepare_family_ticket_frequencies(x_train, x_test, train)
    print(f'Debug time: Family ticket frequencies: {time.time() - time_stamp} seconds')

    # 10 --> Age - fill in missing values, bin
    time_stamp = time.time()
    impute_age_regression(x_train, x_test)
    print(f'Debug time: Impute age: {time.time() - time_stamp} seconds')
    if 'Bin' in options['feature_view']['Age']:
        x_train['Age Bin'] = x_train['Age'].apply(manual_age_bin)
        x_test['Age Bin'] = x_test['Age'].apply(manual_age_bin)
        print(f"Debug: EVERY_TIME Age value_counts x_train:\n{x_train['Age Bin'].value_counts().sort_index()}")
        print(f"Debug: EVERY_TIME Age value_counts x_test:\n{x_test['Age Bin'].value_counts().sort_index()}")
        features_to_add_dummies.append('Age Bin')
    if 'Num' not in options['feature_view']['Age']:
        features_to_drop_after_use.append('Age')

    print(f'Debug: EVERY_TIME Features before dropping not used at all, x_train '
          f'shape {x_train.shape}: {x_train.columns.values}')

    x_train.drop(features_to_drop_after_use, axis=1, inplace=True)
    x_test.drop(features_to_drop_after_use, axis=1, inplace=True)

    print(f'Debug: EVERY_TIME Features before removing features saved for imputing: x_train '
          f'shape {x_train.shape}: {x_train.columns.values}')

    features_saved_for_imputing = [feat for feat in x_train.columns.values if 'IMPUTE' in feat]

    x_train.drop(features_saved_for_imputing, axis=1, inplace=True)
    x_test.drop(features_saved_for_imputing, axis=1, inplace=True)

    print(f'Debug: EVERY_TIME Features before major dropping, x_train '
          f'shape {x_train.shape}: {x_train.columns.values}')

    x_train.drop(options['major_columns_to_drop_every_time'], axis=1, inplace=True)
    x_test.drop(options['major_columns_to_drop_every_time'], axis=1, inplace=True)

    print(f'Debug: EVERY_TIME Features after dropping major, before dummies, x_train shape {x_train.shape}: '
          f'{x_train.columns.values}')

    x_train = pd.get_dummies(x_train, columns=features_to_add_dummies)
    x_test = pd.get_dummies(x_test, columns=features_to_add_dummies)

    print(f'Debug: EVERY_TIME Features after dummies before dropping minor, x_train shape {x_train.shape}: '
          f'{x_train.columns.values}')

    x_train.drop(options['minor_columns_to_drop_every_time'], axis=1, inplace=True)
    x_test.drop(options['minor_columns_to_drop_every_time'], axis=1, inplace=True)

    print(f'Debug: EVERY_TIME Features after dummies after dropping minor, x_train shape {x_train.shape}: '
          f'{x_train.columns.values}')

    print(f'Debug: EVERY_TIME x_train.info:\n{x_train.info()}')
    print(f'Debug: EVERY_TIME Value counts of all values x_train:')
    for feat in x_train.columns.values:
        print(f'--------------- {feat}:')
        print(x_train[feat].value_counts())

    # TODO - should it be here?
    # TODO used to print to file. Can still print to file if correlations are different multiple times due to different data?
    #   File will be overridden? Doesn't make sense to save per permutation.  Perhaps only to print
    # TODO make sure all data is printed properly and seen on the display
    print(x_train.corr())

    scaler = StandardScaler()
    scaler.fit(x_train)

    return scaler.transform(x_train), scaler.transform(x_test), x_train.columns


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
        print(f'Debug doing fold {cv_fold} of {name_str} of type {type_class}')
        fit_detailed(name_str, type_class, classifier, x_train, y_train, x_test, train,
                     results, preds, train_probas, test_probas, start_time, cv_fold)


def get_cross_val_score_various_rands(classifier, x_train, y_train, cv_folds, num_rands):
    accuracies_with_rand = []

    full_train = pd.concat([x_train, y_train], axis=1)

    for rand_loop in range(num_rands):
        rand = rand_loop + RANDOM_STATE_FIRST
        classifier.set_params(random_state=rand_loop + RANDOM_STATE_FIRST)
        print(f'Debug: doing rand {rand_loop + 1} out of {num_rands}')

        fold = KFold(cv_folds, True, random_state=rand)
        i = 0

        for train_indicies, test_indicies in fold.split(x_train, y_train):
            i = i + 1
            print(f'Debug: doing fold {i} out of {cv_folds}')
            train = full_train.iloc[train_indicies]
            time_temp = time.time()
            # Copy is done to prevent changes to original data frame since numerous different changes will be done
            x_train_scaled, x_test_scaled, columns = prepare_features_scale_every_time(
                x_train.iloc[train_indicies].copy(), x_train.iloc[test_indicies].copy(), train)
            print(f'Debug time: Preparing features: {time.time() - time_temp} seconds')
            classifier.fit(x_train_scaled, y_train.reset_index(drop=True)[train_indicies])
            accuracy = classifier.score(x_test_scaled, y_train.reset_index(drop=True)[test_indicies])
            accuracies_with_rand.append(accuracy)

    return np.mean(accuracies_with_rand), np.std(accuracies_with_rand)


def fit_detailed(name_str, type_class, classifier, x_train, y_train, x_test, train, results, preds,
                 train_probas, test_probas, start_time, cv_folds):

    cross_acc_score, cross_acc_std = get_cross_val_score_various_rands(
        classifier, x_train, y_train, cv_folds, options['num_rands'])

    x_train_scaled, x_test_scaled, columns = prepare_features_scale_every_time(
        x_train.copy(), x_test.copy(), train)

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

    results.append({'Features options': str(options['feature_view']),
                    'Name': name_str,
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


def single_features_view(x_train, y_train, x_test, train, results, output_folder, feature_view_name):
    start_time_total = time.time()

    x_train = prepare_features_scale_once(x_train, train)
    x_test = prepare_features_scale_once(x_test, train)
    train = pd.concat([x_train, y_train], axis=1)

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

    view_output_folder = output_folder + '/' + feature_view_name + '/'
    os.mkdir(view_output_folder)

    preds.corr().to_csv(view_output_folder + 'classifiers_correlations.csv')

    if options['output_preds']:
        output_all_preds(preds, x_test, view_output_folder)

    print(f'Debug: single_features_view Time took: {time.time() - start_time_total} seconds = '
          f'{round((time.time() - start_time_total) / 60)} minutes ')


def main():
    start_time_total = time.time()
    options['feature_view'] = {}
    results = []

    output_folder = 'output/_' + time.strftime("%Y_%m_%d_%H_%M_%S") + '/'
    os.mkdir(output_folder)

    write_to_file_input_options(output_folder)

    train, x_test = read_files()
    x_train = train.drop('Survived', axis=1)
    y_train = train['Survived']

    for fam_opt in options['features_variations']['Family size']:
        options['feature_view']['Family size'] = fam_opt
        for fare_opt in options['features_variations']['Fare']:
            options['feature_view']['Fare'] = fare_opt
            for age_opt in options['features_variations']['Age']:
                options['feature_view']['Age'] = age_opt
                single_features_view(x_train.copy(), y_train.copy(), x_test.copy(), train.copy(), results,
                                     output_folder, 'Fam_' + fam_opt + '_Fare_' + fare_opt + '_Age_' + age_opt)

    pd.DataFrame(results).to_csv(output_folder + 'results.csv')

    print(f'Debug: Total Time took: {time.time() - start_time_total} seconds = '
          f'{round((time.time() - start_time_total) / 60)} minutes ')


'''
TODO:
Beginning:
- k-Fold actual training (in addition to Bagging? Instead?) How to actually combine results?
- Consider Bagging and not just cross-validation at one of the lower levels
    Use Out of Bag accuracy when doing Bagging
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
- Consider adding more views of features (age, family size etc.)
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
    'cv_folds': [5],  # options of number of folds for Cross validation.  Try with 10 also - gives even worse result
    'num_rands': 5,  # number of times to run the same thing with various random numbers. Better to have 10-15, 5 for now to make it quicker
    # TODO is there a nice way to do it than to split up both major and minor into once and every time?
    # main columns to drop
    'major_columns_to_drop_once': [
        'Ticket',  # not helpful as is, but could have been used in feature extraction
        'Name',  # not helpful as is, but could have been used in feature extraction
        'Cabin',  # not helpful as is, but could have been used in feature extraction
        'Sex',  # Since titles are important, need to remove Sex
        'SibSp',  # very low in all models, perhaps because of Family size / Ticket_Frequency
        'Parch',  # very low in all models, perhaps because of Family size / Ticket_Frequency
        'Embarked',
    ],
    'major_columns_to_drop_every_time': [
        'Family/ticket survival known',  # low in all models
    ],
    # specific binned features to drop, like a specific bin in the main feature
    'minor_columns_to_drop_once': [
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
    # specific binned features to drop, like a specific bin in the main feature
    'minor_columns_to_drop_every_time': [
        # -- Age - not extemely important, most models Age_-4 is important (15), XGB gives more age importance (6,8)
        # 'Age Bin_-4',
        # TODO now was not commented out, but since one of the options is not to have bins at all, had to comment out
        #   will not be relevant, once automatically remove not important features
        # 'Age Bin_4-11',  # low in all classifiers
        # 'Age Bin_11-24',
        # 'Age Bin_24-32',
        # 'Age Bin_32-42',
        # 'Age Bin_42+'
    ],
    'features_variations': {
        # 'SibSp': ['Num', 'ManualBin', 'AutobinXXX', 'Num+ManualBin'],  # TODO - add - currently not used
        # 'Parch': ['Num', 'ManualBin', 'AutobinXXX', 'Num+ManualBin'],  # TODO - add - currently not used

        'Family size': ['Bin', 'Num', 'Both'],  # TODO - add a different way to bin?
        # 'Family size': ['Bin'],  # TODO - add a different way to bin?
        'Age': ['Bin', 'Num', 'Bin+Num'],  # TODO - add other ways to bin?
        'Fare': ['Num', 'Log', '13.5+', 'Log+13.5', 'Num+13.5'],  # TODO - should add more ways, like manual or automatic bin?
        # 'Fare': ['13.5+'],  # TODO - should add more ways, like manual or automatic bin?

        # 'Deck': []  # TODO - should add different ways to combine?
    },
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

main()
