import time
import numpy as np
import pandas as pd
import seaborn as sns
import os
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
sns.set()

NUM_TRAIN_SAMPLES = 891
RANDOM_STATE = 50


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
    print(f"Debug: Age prediction feature importance, features: {features_base_age_on}, "
          f"importance:\n{age_model.feature_importances_}")
    age_preds = age_model.predict(x_test)

    both.loc[both['Age'].isnull(), 'Age'] = age_preds

    print(f'Debug: Age prediction score: {round(age_model.score(x_train, y_train) * 100,1)}')
    return age_preds


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

    print(f'Debug: Train survival rates: \n'
          f'{split_into_x_train_x_test(both)[0]["Known family/ticket survived %"].value_counts(dropna=False)}')
    print(f'Debug: Test survival rates: \n'
          f'{split_into_x_train_x_test(both)[1]["Known family/ticket survived %"].value_counts(dropna=False)}')

    # drop temporary columns used
    both.drop(['Last name', 'Known family survived %', 'Known ticket survived %'], axis=1, inplace=True)


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


def prepare_features(train, x_test, options, output_folder):
    num_train_samples = train.shape[0]

    print(f'Debug: RANDOM_STATE:{RANDOM_STATE}, num_train_samples:{num_train_samples}')
    print(f'Debug: Features before adding / dropping: {x_test.columns.values}')

    features_to_drop_after_use = []
    features_to_add_dummies = []
    both = combine_train_x_test(train, x_test)

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
    prepare_family_ticket_frequencies(both, train['Survived'])

    # 10 --> Age - fill in missing values, bin
    impute_age_regression(both)
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

    return new_x_train, new_x_test


def output_single_preds(preds, x_test, output_folder, name_suffix):
    pred_df = pd.DataFrame(preds)
    pred_df.set_index(x_test.index, inplace=True)
    pred_df.columns = ['Survived']
    pred_df.to_csv(f'{output_folder}preds_{name_suffix}.csv')


def output_all_preds(preds, x_test, output_folder):
    output_single_preds(preds['RF 7'], x_test, output_folder, 'rf_7')

    output_single_preds(preds['Grid Log'], x_test, output_folder, 'log_grid')
    output_single_preds(preds['Grid KNN'], x_test, output_folder, 'knn_grid')
    output_single_preds(preds['Grid SVM'], x_test, output_folder, 'svm_grid')
    output_single_preds(preds['Grid RF'], x_test, output_folder, 'rf_grid')
    output_single_preds(preds['Grid XGB'], x_test, output_folder, 'xgb_grid')

    output_single_preds(preds['Voting soft - part of grid'], x_test, output_folder, 'voting_soft')
    output_single_preds(preds['Voting hard - part of grid'], x_test, output_folder, 'voting_hard')

    output_single_preds(preds['Ensemble RF - part of grid'], x_test, output_folder, 'ensemble_rf')
    output_single_preds(preds['Ensemble Log - part of grid'], x_test, output_folder, 'ensemble_log')


def cross_valid(classifier, x_train, y_train):
    accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=5)
    return accuracies.mean(), accuracies.std()


def fit_different_classifiers(name_str, type_class, classifier, x_train, y_train, x_test, results, preds,
                              train_probas, test_probas, start_time):
    reg_score, reg_std = cross_valid(classifier, x_train, y_train)

    classifier.fit(x_train, y_train)
    preds[name_str] = classifier.predict(x_test)
    score = classifier.score(x_train, y_train)

    try:
        train_probas[name_str] = classifier.predict_proba(x_train)[:, 0]
        test_probas[name_str] = classifier.predict_proba(x_test)[:, 0]
    except AttributeError:
        # For Hard voting probabilities where predict_proba is not supported
        train_probas[name_str] = np.mean(x_train, axis=1)
        test_probas[name_str] = np.mean(x_test, axis=1)

    results.append({'Name': name_str,
                    'Single accuracy': round(score, 3),
                    'Cross accuracy': round(reg_score, 3),
                    'STD': round(reg_std, 3),
                    'Cross accuracy-STD*2': round(reg_score - reg_std * 2, 3),
                    'Cross accuracy-STD*3': round(reg_score - reg_std * 3, 3),
                    'Overfitting danger': round((score - reg_score) * score, 3),
                    'Time sec': round(time.time() - start_time)})
    print(f'Debug: Stats {type_class}: {results[-1]}')


def fit_grid_classifier(name_str, x_train, y_train, x_test, single_classifier, grid_params, results, preds,
                        train_probas, test_probas):
    start_time = time.time()

    grid = GridSearchCV(single_classifier, grid_params, verbose=1, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)
    classifier = grid.best_estimator_
    print(f'Debug: {name_str} best classifier:\n{classifier}')

    fit_different_classifiers(name_str, 'Grid', classifier, x_train, y_train, x_test, results, preds,
                              train_probas, test_probas, start_time)

    return classifier


def fit_single_classifier(name_str, x_train, y_train, x_test, classifier, results, preds, train_probas, test_probas):
    start_time = time.time()

    fit_different_classifiers(name_str, 'Single', classifier, x_train, y_train, x_test, results, preds,
                              train_probas, test_probas, start_time)

    return classifier


def fit_predict_voting(classifiers, name_str, voting_type, x_train, y_train, x_test, results, preds,
                       train_probas, test_probas):
    start_time = time.time()

    classifier = VotingClassifier(estimators=classifiers, voting=voting_type, n_jobs=-1)

    fit_different_classifiers(name_str, 'Voting', classifier, x_train, y_train, x_test, results, preds,
                              train_probas, test_probas, start_time)

    return classifier


def print_feature_importances(cl, classifier, x_train):
    if cl == 'Log':
        importances = pd.DataFrame({'Importance': classifier.coef_[0]}, index=x_train.columns). \
            reset_index().sort_values(by='Importance', ascending=False)
        print(f'Debug: "{cl}" feature importances:\n{pd.DataFrame(importances)}')
        importances['Importance'] = importances['Importance'].abs()
        print(f'Debug: "{cl}" feature importances (abs):\n'
              f'{pd.DataFrame(importances).sort_values(by="Importance", ascending=False).reset_index()}')
    elif 'RF' in cl:
        importances = pd.DataFrame({'Importance': classifier.feature_importances_}, index=x_train.columns).\
            reset_index().sort_values(by='Importance', ascending=False).reset_index()
        print(f'Debug: "{cl}" feature importances:\n{importances}')
    elif cl == 'XGB':
        importance = pd.DataFrame(classifier.get_booster().get_score(importance_type="gain"),
                                  index=["Importance"]).transpose()
        print(f'Debug: "{cl}" feature importances:\n'
              f'{importance.sort_values(by="Importance", ascending=False).reset_index()}')


def write_to_file_input_options(output_folder, options):
    w = csv.writer(open(output_folder + 'input_options.csv', 'w', newline=''))
    for key, val in options.items():
        if key not in options['input_options_not_to_output']:
            w.writerow([key, val])


def main(options):
    start_time_total = time.time()

    output_folder = 'output/' + time.strftime("%Y_%m_%d_%H_%M_%S") + '/'
    os.mkdir(output_folder)

    write_to_file_input_options(output_folder, options)

    train, x_test = read_files()

    x_train, x_test = prepare_features(train, x_test, options, output_folder)

    y_train = train['Survived']

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

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
                                           x_train_scaled,
                                           y_train,
                                           x_test_scaled,
                                           single_classifiers[cl]['clas'],
                                           results, preds, unused_train_proba, unused_test_proba)
        # Currently, only using Grid classifiers for voting
        '''
        if cl not in grid_classifiers_not_for_ensembling:
            classifiers_for_ensembling.append((cl, classifier))
        '''
        # print feature importances for classifiers where it's easy to get this information
        print_feature_importances(cl, classifier, x_train)

    for cl in grid_classifiers:
        classifier = fit_grid_classifier(
            cl,
            x_train_scaled,
            y_train,
            x_test_scaled,
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

    fit_predict_voting(classifiers_for_ensembling, 'Voting soft - part of grid', 'soft',
                       x_train_scaled, y_train, x_test_scaled,
                       results, preds, unused_train_proba, unused_test_proba)
    fit_predict_voting(classifiers_for_ensembling, 'Voting hard - part of grid', 'hard',
                       x_train_scaled, y_train, x_test_scaled,
                       results, preds, unused_train_proba, unused_test_proba)

    # Ensembling based on probabilities of previous classifiers
    # Based on part of the Grid results - see grid_classifiers_not_for_ensembling

    print(f'Debug: shape of train_probas: {train_probas.shape}, test_probas: {test_probas.shape}')
    print(f'Debug: head of train_probas:\n{train_probas.head()}')

    fit_grid_classifier('Ensemble RF - part of grid', train_probas, y_train, test_probas,
                        RandomForestClassifier(n_estimators=1000, random_state=RANDOM_STATE, n_jobs=-1),
                        [{'max_depth': range(3, 10)}],
                        results, preds, unused_train_proba, unused_test_proba)

    fit_grid_classifier('Ensemble Log - part of grid', train_probas, y_train, test_probas,
                        LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, n_jobs=-1),
                        [{'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']}],
                        results, preds, unused_train_proba, unused_test_proba)

    preds.corr().to_csv(output_folder + 'classifiers_correlations.csv')

    output_all_preds(preds, x_test, output_folder)

    pd.DataFrame(results).to_csv(output_folder + 'results.csv')

    print(f'Debug: Time took: {time.time() - start_time_total} seconds = '
          f'{round((time.time() - start_time_total) / 60)} minutes ')


'''
TODO:
Beginning:
- Use different scores: cross_val_score(model, X, Y, cv=kfold, scoring=<method>). 
    Confusion Matrix / Precision / Recall / F1 Score, or ROC curve
- Do different views of the features (what's included / not included / in what format)
- Shuffle with random_state - some algorithms are effected by the order
A bit later:
- Take code and ideas from https://machinelearningmastery.com/spot-check-machine-learning-algorithms-in-python/
- Do all feature preparation on each fold separately - both train and test, and each fold of the train.  This will prevent leakage, but it will actually probably lower the score
- Look only at - STD*3
- First do Grid, then cross-validation
- Introduce random state for cross-validation if not used today
- Consider Bagging and not just cross-validation at one of the lower levels
    Use Out of Bag accuracy when doing Bagging
- k-Fold actual training (in addition to Bagging? Instead?) How to actually combine results?
    kfold = KFold(n_splits=10, random_state=7)
    model = LogisticRegression(solver='liblinear')
    results = cross_val_score(model, X, Y, cv=kfold)
- Play with cross-validation size - give a few
- Play with an average of a few random sizes 
- Automate bottom line report and choosing of the model
- Do feature selection with RFECV per algorithm 
Middle:
- Add extra trees algorithm, AdaBoost, Bernoulli NB (perhaps instead / in addition to Gaussasian NB), others from his list of best / all
    From his list of algorithms for classification: Random Forest, XGBoost, SVM, (Backpropogation - what specifically is it?), Decision Trees (CART and C4.5/C5.0), Naive Bayes, Logistic Regression and Linear Discriminant Analysis, k-Nearest Neighbors and Learning Vector Quantization (what is it?)
- Give a chance to each one of the classifiers
- XGBoost - do much more parameter optimizations
End:
- Voting only on models I know work best
- Consider using statistical tests to decide with algorithm is better: parametric / non parametric, P-value

'''

options = {
    'input_options_not_to_output': ['single_classifiers', 'grid_classifiers'],
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
        'Log': {'clas': LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, n_jobs=-1)},
        'KNN 14': {'clas': KNeighborsClassifier(n_neighbors=14, n_jobs=-1)},
        'SVM rbf': {'clas': SVC(gamma='auto', kernel='rbf', probability=True, random_state=RANDOM_STATE)},
        'SVM poly': {'clas': SVC(gamma='auto', kernel='poly', probability=True, random_state=RANDOM_STATE)},
        'NB': {'clas': GaussianNB()},  # consistently gives worse results
        'RF 10': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)},
        'RF 9': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=9, random_state=RANDOM_STATE, n_jobs=-1)},
        'RF 8': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)},
        'RF 7': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=RANDOM_STATE, n_jobs=-1)},
        'RF 6': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)},
        'XGB': {'clas': xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000,
                                          random_state=RANDOM_STATE, n_jobs=-1)}},
    # Classifiers we use with Grid search
    'grid_classifiers': {
        'Grid Log': {'clas': LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, n_jobs=-1),
                     'grid_params': [{'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']}]},
        'Grid KNN': {'clas': KNeighborsClassifier(n_neighbors=14, n_jobs=-1),
                     'grid_params': [{'n_neighbors': range(3, 25)}]},
        'Grid SVM': {'clas': SVC(gamma='auto', kernel='rbf', probability=True, random_state=RANDOM_STATE),
                     'grid_params':
                         [{
                            'kernel': ['rbf', 'poly', 'sigmoid'],
                            'C': [0.3, 0.5, 1.0, 1.5, 2.0],
                            'gamma': [0.3, 0.2, 0.1, 0.05, 0.01, 'auto_deprecated', 'scale']
                         }],
                     },
        'Grid RF': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=RANDOM_STATE, n_jobs=-1),
                    'grid_params': [{'max_depth': range(3, 10)}]},
        'Grid XGB': {'clas': xgb.XGBClassifier(objective='binary:logistic',
                                               n_estimators=1000,
                                               random_state=RANDOM_STATE,
                                               n_jobs=-1),
                     'grid_params':
                         [{
                             'max_depth': range(1, 8, 1)  # default 3 - higher depth - less bias, more variance
                             # 'n_estimators': range(60, 260, 40), # default 100
                             # 'learning_rate': [0.3, 0.2, 0.1, 0.01],  # , 0.001, 0.0001
                             # 'min_child_weight': [0.5, 1, 2],  # default 1 - higher number, less overfitting, when to stop splitting the child given sum of weights
                             # 'subsample': [i / 10.0 for i in range(6, 11)], # default 1, smaller values prevent overfitting
                             # 'colsample_bytree': [i / 10.0 for i in range(6, 11)] # default 1, fraction of features selected for each tree
                             # 'gamma': [i / 10.0 for i in range(3)]  # default 0 - for what gain in metric to continue splitting
                         }]
                     }
    },
    'grid_classifiers_not_for_ensembling': ['Grid SVM', 'Grid XGB']
}

main(options)
