## Solution of titanic dataset from Kaggle
Link to Kaggle: https://www.kaggle.com/c/titanic

## Main files
- Investigation notebooks - in main directory. Order advised to go through:
    - `initial-titanic-logistic-regression.ipynb`
    - `Fare_investigation.ipynb`
    - `age_investigation.ipynb`
    - `more_feature_engineering.ipynb`
    - `Advanced feature engineering.ipynb`
- `titanic.py` - main and only code file that does all steps from preprocessing to producing predictions
- `input` folder:
  - `train.csv` - training input file with expected label 
  - `test.csv` -  test input file
  - `test_answers.csv` - True North.  Not used not to create leakage, left here for completeness
- `output` folder: **outputs of last execution, changes per execution**
  - `classifiers_correlations.csv` - correlations between predictions of different classifiers' used
  - `feature_correlations.csv` - correlations between features used
  - `results.csv` - results of all classifiers used: 
     - cross validation accuracy
     - cross validation STD
     ...
  - `pred...` files - output of predictions of specific classifier
    - If has `grid` in the name - result of doing Grid Search on the classifier
    - if has `voting` in the name - result of doing Voting (Soft, Hard) on results of previous classifiers
    - If has `ensemble` in the name - result of doing ensembling based on probabilities produced by single classifiers
    
## Classifiers used
- `LogisticRegression`
- `KNeighborsClassifier`
- `SVC` - classifier based on Support Vector Machines
- `GaussianNB` - Naive Bayes
- `RandomForestClassifier`
- `XGBClassifier` from xgboost library
** For most classifiers, Grid Search was done to find the best hyperparameters**
  
## Ensembling done:
- Soft voting
- Hard voting
- `LogisticRegression` ensembling based on output of probabilities of single classifiers
- `RandomForestClassifier` ensembling based on output of probabilities of single classifiers

## Results:
  - Locally accuracy measured for different classifiers with cross validation ranges around ~85% with STD of 1-2% 
  - On Kaggle site, most results are round ~80% accuracy, with soft voting getting slightly more.  
    Hard voting, logistic regression, KNN, and Random Forest all get slightly under ~80%

## Features used:
  - `Title` - retrieved from `Name` feature, and grouped into cagetories, then added dummies
  - `Deck` - retrieved from first letter of `Cabin` feature, and grouped into cagetories, then added dummies. 
    Only categories that helped the model were left.
  - `Pclass`
  - `Ticket_Frequency` - number of passengers on the same ticket (slight leakage due to taking also from test set)
  - `Fare` per person of > 13.5 (needed to divide `Fare` by `Ticket_Frequency`). Other categories and splits, using as is, or doing `Log` on the numbers didn't help the model
  - `Age` - binned by survival rate.  See about imputing below
  - `Known family/ticket survived %` - look both based on family (last name) and ticket, and add average % of survived in training set.
 
 ## Features not used:
 - `Family size` - sum of `` + `` + 1 - didn't help the model, perhaps because of high correlation with `Ticket_Frequency` above  
 - `SibSp` - same
 - `Parch` - same
 - `Sex` - didn't help the model, perhaps because of high correlation with `Title` above  
 - `Embarked` - didn't help the model
  **Many other features were used that were later removed because didn't help the model or had very high correlation with other features**
  
## More notes:
- Imputing was done on 'Age' for missing values with `RandomForestRegressor` using other features
- Scaling was done on all features
