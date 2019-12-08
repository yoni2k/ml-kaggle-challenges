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
RANDOM_STATE = 50

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
        return '13.5+'


def manual_age_bin(age):
    if age <= 4:
        return '-4'  # based on survival rate
    elif age <= 11:
        return '4-11'  # based on survival rate
    elif age <= 24:
        return '11-24'  # based on survival rate
    elif age <= 26:
        return '24-26'  # based on survival rate
    elif age <= 27:
        return '26-27'  # based on survival rate
    elif age <= 31:
        return '27-31'  # based on survival rate
    elif age <= 32:
        return '31-32'  # based on survival rate
    elif age <= 40:
        return '32-40'  # based on KDE
    elif age <= 48:
        return '40-48'  # based on KDE
    elif age <= 57:
        return '48-57'  # based on KDE
    else:
        return '57+'  # based on KDE


def prepare_features(train, x_test, options):
    num_train_samples = train.shape[0]

    print(f'YK: RANDOM_STATE:{RANDOM_STATE}')
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

    both['Family size'] = both['Family size'].replace({1: '1',
                                                       2: '23', 3: '23',
                                                       4: '4',
                                                       5: '567', 6: '567', 7: '567',
                                                       8: '8+', 11: '8+'})
    features_to_add_dummies.append('Family size')


    # 3. ----> Prepare Deck features based on first letter of Cabin, unknown Cabin becomes reference category
    # Reference category is being removed later based on importance of each of the categories
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
    # TODO decide if to leave 'Fare bin' categorical, or turn into numerical
    both['Fare bin'] = both['Fare per person'].apply(manual_fare_bin_categorical)
    both['Fare log'] = both['Fare per person'].replace({0: 0.0001}) # to avoid doing log on 0 which is invalid
    both['Fare log'] = np.log(both['Fare log'])
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
    both['Age Bin'] = both['Age'].apply(manual_age_bin)
    print(f"Age value_counts:\n{both['Age Bin'].value_counts().sort_index()}")
    features_to_add_dummies.append('Age Bin')

    both.drop(features_to_drop_after_use, axis=1, inplace=True)

    print(f'YK: Features after dropping not used at all, before major dropping, shape {both.shape}: {both.columns.values}')

    both.drop(options['major_columns_to_drop'], axis=1, inplace=True)

    print(f'YK: Features after dropping major, before dummies, shape {both.shape}: {both.columns.values}')

    both = pd.get_dummies(both, columns=features_to_add_dummies)

    print(f'YK: Features after dummies before dropping minor, shape {both.shape}: {both.columns.values}')

    both.drop(options['minor_columns_to_drop'], axis=1, inplace=True)

    print(f'YK: Features after dummies after dropping minor, shape {both.shape}: {both.columns.values}')

    print(f'YK: both.info():\n{both.info()}')
    print(f'YK: Value counts of all values:')
    for feat in both.columns.values:
        print(f'--------------- {feat}:')
        print(both[feat].value_counts())

    both.corr().to_csv('output/feature_correlations.csv')

    new_x_train, new_x_test = split_into_x_train_x_test(both)

    return new_x_train.drop(options['features_to_drop_continous'], axis=1), \
           new_x_test.drop(options['features_to_drop_continous'], axis=1), \
           new_x_train.drop(options['features_to_drop_forest'], axis=1), \
           new_x_test.drop(options['features_to_drop_forest'], axis=1)


def scale_train(x_train):
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler


def output_preds(preds, x_test, name_suffix):
    pred_df = pd.DataFrame(preds)
    pred_df.set_index(x_test.index, inplace=True)
    pred_df.columns = ['Survived']
    pred_df.to_csv(f'output/preds_{name_suffix}.csv')


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
    print(f'Stats {type_class}: {results[-1]}')


def fit_grid_classifier(name_str, x_train, y_train, x_test, single_classifier, grid_params, results, preds,
                        train_probas, test_probas):
    start_time = time.time()

    grid = GridSearchCV(single_classifier, grid_params, verbose=1, cv=5, n_jobs=-1)
    grid.fit(x_train, y_train)
    classifier = grid.best_estimator_
    print(f'{name_str} best classifier:\n{classifier}')

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


def fit_ensemble(name_str, train_probas, test_probas, y_train, results, preds):
    start_time = time.time()

    classifier = RandomForestClassifier(n_estimators=1000, random_state=RANDOM_STATE, n_jobs=-1)

    fit_different_classifiers(name_str, 'Emsemble', classifier, train_probas, y_train, test_probas, results, preds,
                              train_probas, test_probas, start_time)



def main(options):
    start_time_total = time.time()

    train, x_test = read_files()

    x_train_cont, x_test_cont, x_train_not_cont, x_test_not_cont = prepare_features(train, x_test, options)

    y_train = train['Survived']

    scaler = scale_train(x_train_cont)
    x_train_scaled_cont = scaler.transform(x_train_cont)
    x_test_scaled_cont = scaler.transform(x_test_cont)

    scaler = scale_train(x_train_not_cont)
    x_train_scaled_not_cont = scaler.transform(x_train_not_cont)
    x_test_scaled_not_cont = scaler.transform(x_test_not_cont)

    results = []
    preds = pd.DataFrame()
    train_probas = pd.DataFrame()
    test_probas = pd.DataFrame()

    single_classifiers = {
        #'Log': {'clas': LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, n_jobs=-1),
        #        'Continuous': True},
        #'KNN 14': {'clas': KNeighborsClassifier(n_neighbors=14, n_jobs=-1),
        #   'Continuous': True},
        #'SVM rbf': {'clas': SVC(gamma='auto', kernel='rbf', probability=True, random_state=RANDOM_STATE),
        #            'Continuous': True},
        #'SVM poly': {'clas': SVC(gamma='auto', kernel='poly', probability=True, random_state=RANDOM_STATE),
        #             'Continuous': True},
        # 'NB': {'clas': GaussianNB(), 'Continuous': True},  # consistently gives worse results
        #'RF 9': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=9, random_state=RANDOM_STATE, n_jobs=-1),
        #         'Continuous': False},
        #'RF 8': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1),
        #         'Continuous': False},
        #'RF 7': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=RANDOM_STATE, n_jobs=-1),
        #         'Continuous': False},
        #'RF 6': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1),
        #        'Continuous': False},
        'XGB': {'clas': xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000,
                                          random_state=RANDOM_STATE, n_jobs=-1),
                'Continuous': True}
    }

    classifier_not_for_soft = []  # was 'XGB', 'Grid XGB'
    classifier_not_for_hard = ['NB', 'Log', 'SVM poly', 'Grid KNN']
    classifiers_for_voting_soft = []
    classifiers_for_voting_hard = []

    for cl in single_classifiers:
        if single_classifiers[cl]['Continuous']:
            classifier = fit_single_classifier(cl,
                                               x_train_scaled_cont,
                                               y_train,
                                               x_test_scaled_cont,
                                               single_classifiers[cl]['clas'],
                                               results, preds, train_probas, test_probas)
            # Since continuous and not continuous don't use the same number of features, can't do voting together
            if cl not in classifier_not_for_soft:
                classifiers_for_voting_soft.append((cl, classifier))
            if cl not in classifier_not_for_hard:
                classifiers_for_voting_hard.append((cl, classifier))
        else:
            classifier = fit_single_classifier(cl,
                                               x_train_scaled_not_cont,
                                               y_train,
                                               x_test_scaled_not_cont,
                                               single_classifiers[cl]['clas'],
                                               results, preds, train_probas, test_probas)

        if cl == 'Log':
            importances = pd.DataFrame({'Importance': classifier.coef_[0]}, index=x_train_cont.columns). \
                reset_index().sort_values(by='Importance', ascending=False)
            print(f'YK: "{cl}" feature importances:\n{pd.DataFrame(importances)}')
            importances['Importance'] = importances['Importance'].abs()
            print(f'YK: "{cl}" feature importances (abs):\n{pd.DataFrame(importances).sort_values(by="Importance", ascending=False).reset_index()}')
        elif 'RF' in cl:
            importances = pd.DataFrame({'Importance': classifier.feature_importances_}, index=x_train_not_cont.columns).\
                reset_index().sort_values(by='Importance', ascending=False).reset_index()
            print(f'YK: "{cl}" feature importances:\n{importances}')
        elif cl == 'XGB':
            importance = pd.DataFrame(classifier.get_booster().get_score(importance_type="gain"),
                                      index=["Importance"]).transpose()
            print(f'YK: "{cl}" feature importances:\n{importance.sort_values(by="Importance", ascending=False).reset_index()}')

    # TODO keep?
    #grid_classifiers = {
    #    'Grid Log': {'clas': LogisticRegression(solver='liblinear', n_jobs=-1, random_state=RANDOM_STATE),
    #                 'grid_params': [{'solver': ['liblinear', 'lbfgs']}]}}

    grid_classifiers = {
        #'Grid Log': {'clas': LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, n_jobs=-1),
        #             'grid_params': [{'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']}]},
        #'Grid KNN': {'clas': KNeighborsClassifier(n_neighbors=14, n_jobs=-1),
        #             'grid_params': [{'n_neighbors': range(3, 25)}],
        #             'Continuous': True},
        'Grid SVM': {'clas': SVC(gamma='auto', kernel='rbf', probability=True, random_state=RANDOM_STATE),
                     'grid_params':
                         [{
                            'kernel': ['rbf', 'poly', 'sigmoid'],
                            'C': [0.3, 0.5, 1.0, 1.5, 2.0],
                            'gamma': [0.3, 0.2, 0.1, 0.05, 0.01, 'auto_deprecated', 'scale']
                         }],
                     'Continuous': True
                     },
        'Grid RF': {'clas': RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=RANDOM_STATE,n_jobs=-1),
                    'grid_params': [{'max_depth': range(3, 10)}],
                    'Continuous': False},
        #'Grid XGB': {'clas': xgb.XGBClassifier(objective='binary:logistic', n_estimators=1000, random_state=RANDOM_STATE, n_jobs=-1),
        #            'grid_params':
        #                 [{
        #                     'max_depth': range(1, 8, 1)  # default 3
        #                     # 'n_estimators': range(60, 260, 40), # default 100
        #                     # 'learning_rate': [0.3, 0.2, 0.1, 0.01],  # , 0.001, 0.0001
        #                     # 'min_child_weight': [0.5, 1, 2],  # default 1
        #                     # 'subsample': [i / 10.0 for i in range(6, 11)], # default 1, not sure needed
        #                     # 'colsample_bytree': [i / 10.0 for i in range(6, 11)] # default 1, not sure needed
        #                     # 'gamma': [i / 10.0 for i in range(3)]  # default 0
        #                 }],
        #             'Continuous': True
         #            }
    }

    for cl in grid_classifiers:
        if grid_classifiers[cl]['Continuous']:
            classifier = fit_grid_classifier(cl,
                                             x_train_scaled_cont,
                                             y_train,
                                             x_test_scaled_cont,
                                             grid_classifiers[cl]['clas'], grid_classifiers[cl]['grid_params'],
                                             results, preds, train_probas, test_probas)
            # Since continuous and not continuous don't use the same number of features, can't do voting together
            if cl not in classifier_not_for_soft:
                classifiers_for_voting_soft.append((cl, classifier))
            if cl not in classifier_not_for_hard:
                classifiers_for_voting_hard.append((cl, classifier))
        else:
            classifier = fit_grid_classifier(cl,
                                             x_train_scaled_not_cont,
                                             y_train,
                                             x_test_scaled_not_cont,
                                             grid_classifiers[cl]['clas'], grid_classifiers[cl]['grid_params'],
                                             results, preds, train_probas, test_probas)


    fit_predict_voting(classifiers_for_voting_soft, 'Voting soft with grid', 'soft',
                               x_train_scaled_cont, y_train, x_test_scaled_cont,
                               results, preds, train_probas, test_probas)
    fit_predict_voting(classifiers_for_voting_hard, 'Voting hard with grid', 'hard',
                               x_train_scaled_cont, y_train, x_test_scaled_cont,
                               results, preds, train_probas, test_probas)

    fit_ensemble('Ensemble RF - all', train_probas, test_probas, y_train, results, preds)
    fit_ensemble('Ensemble RF - soft RF',
                 train_probas.drop(['XGB', 'Grid SVM', 'Voting hard with grid'], axis=1),
                 test_probas.drop(['XGB', 'Grid SVM', 'Voting hard with grid'], axis=1),
                 y_train, results, preds)
    fit_ensemble('Ensemble RF - hard RF',
                 train_probas.drop(['XGB', 'Grid SVM', 'Voting soft with grid'], axis=1),
                 test_probas.drop(['XGB', 'Grid SVM', 'Voting soft with grid'], axis=1),
                 y_train, results, preds)
    fit_ensemble('Ensemble RF - soft hard RF',
                 train_probas.drop(['XGB', 'Grid SVM'], axis=1),
                 test_probas.drop(['XGB', 'Grid SVM'], axis=1),
                 y_train, results, preds)

    print(f'YK: correlations between predictions:\n{preds.corr()}')
    preds.corr().to_csv('output/classifiers_correlations.csv')

#    output_preds(preds['RF 7'], x_test, 'rf_7')
    output_preds(preds['XGB'], x_test, 'xgb')

    output_preds(preds['Grid SVM'], x_test, 'svm_grid')
    output_preds(preds['Grid RF'], x_test, 'rf_grid')
#    output_preds(preds['Grid XGB'], x_test, 'xgb_grid')

    output_preds(preds['Voting soft with grid'], x_test, 'voting_soft')
    output_preds(preds['Voting hard with grid'], x_test, 'voting_hard')

    output_preds(preds['Ensemble RF - all'], x_test, 'ensemble_all')
    output_preds(preds['Ensemble RF - soft RF'], x_test, 'ensemble_soft')
    output_preds(preds['Ensemble RF - hard RF'], x_test, 'ensemble_hard')
    output_preds(preds['Ensemble RF - soft hard RF'], x_test, 'ensemble_soft_hard')

    pd.DataFrame(results).to_csv('output/results.csv')

    print(f'YK: Time took: {time.time() - start_time_total} seconds = '
          f'{round((time.time() - start_time_total) / 60)} minutes ')

options = {
    'major_columns_to_drop': [
        'Sex',  # Since titles are important, need to remove Sex
        # -- 'Family/ticket survival known'  # low in all 4
        'Family/ticket survival known',
        # -- SibSp/SibSpBin - not extremely important in general (>17 in all models).  Consider removing altogether
        'SibSp',  # very low in all models
        'Parch',  # the only one that has high importance is ParchBin_0, but it has high correlation with Family size_0 anyways, so can remove
        'Embarked'

    ],
    'features_to_drop_continous': [],  # 'Fare bin_13.5+' currently not dropping since get worse results if dropping
    'features_to_drop_forest': ['Fare log', 'Title_Master'],
    'minor_columns_to_drop': [
        # -- Age - not extemely important, most models Age_-4 is important (15), XGB gives more age importance (6,8)
        #       Update 1: Age_-4 is only very important in 1 model, removing another age 'Age_27-31'
        #       Update 2: Age is not extremely important, only 1 model has 8, rest > 15, remove Age_11-24
        # 'Age Bin_-4' - important, very little kids indeed survived the most
        'Age Bin_4-11',  # low in all 4 (perhaps because of titles that serve same purpose)
        'Age Bin_11-24',  # in theory should be important, let the models remove it if it's not
        'Age Bin_24-26',  # WAS NEEDED FOR RF - don't believe it's not overfitting to have this specific age
        'Age Bin_26-27',  # WAS NEEDED FOR RF - don't believe it's not overfitting to have this specific age
        'Age Bin_27-31',
        'Age Bin_31-32',
        'Age Bin_32-40',  # WAS NEEDED FOR RF - don't see anything special about this group, not sure why appeared special
        'Age Bin_40-48',
        'Age Bin_48-57',
        'Age Bin_57+',
        # -- Family size - seems more important than ParchBin and SibSpBin, but less consistent between models:
        #       Update 1: 567 seems important in all by XGB, 1 important in all, 8+ not consistent, Family size_2 low in all
        #       Update 2: important in most models, least important category Family size_3, remove
        # 'Family size_1' - important predictor
        #'Family size_23', # WAS NOT NEEDED FOR RF models removed (perhaps because of Ticket frequency connection), but I think should be returned and let models remove -
        #'Family size_4', # WAS NOT NEEDED FOR RF models removed (perhaps because of Ticket frequency connection), but I think should be returned and let models remove -
        #'Family size_567',
        #'Family size_8+', # WAS NOT NEEDED FOR RF models removed (perhaps because of Ticket frequency connection), but I think should be returned and let models remove -
        # -- Fare bin - mostly not very important
        'Fare bin_0',
        'Fare bin_0.1-4',
        'Fare bin_4-5',
        'Fare bin_5-7',
        'Fare bin_7-7.796',
        'Fare bin_7.796-7.896',
        'Fare bin_7.896-7.925',
        'Fare bin_7.925-8.662',
        'Fare bin_8.662-12.5',
        'Fare bin_12.5-13.5',
        # 'Fare bin_13.5+'  # important, places 2-10


        # -- Deck - some important, some not. what's left is important, unknown_T and DE
        # 'DeckBin_AG',
        # 'DeckBin_B',
        # 'DeckBin_CF',
        # 'DeckBin_DE'  # important, perhaps because of mixed deck and more change for non 1st class to survive
        # 'DeckBin_unknown_T'  # important, especially low survival
        # -- Title - most important in most models: Mr important in all, XGB considers everything besides Mr low. Leaving all
        # 'Title_Master'  # WAS NOT NEEDED FOR RF models removed (only Random Forest) (perhaps because of age connection), but I think should be returned and let models remove -
        # -- Pclass - 3 is most important (1,5,8), 1 second (9,19,22 - perhaps have other proxies), 2 - lowest (12,13,24,38).
        # -- Ticket_Frequency - place 7,10,14, leaving
        # -- Known family/ticket survived % - places 2,4 - one of the most important
    ]
}

main(options)
