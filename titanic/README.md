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
 
 ## Features that explain the most the survival rate:
 *Notes:* 
 - *Mostly based on Logistic Regression and Randon Forest feature importance info*
 - *See notebooks in this folder for much better BI of connections between survival and different features*
 
 **From most important to least important features**: 
 #### Most important features
 - `Title_Mr` - male adult had a very low survival rate
 - `Known family/ticket survived %` - whether close family / people on the same ticket survived has 2nd highest prediction ability
 - `Ticket_Frequency` - proxy for family size, here the relationship is not linear.  Families with 2-4 members had highest survival rates, a person alone had lower survival rate, families with > 4 members had much lower survival rates
 - `Fare 13.5+` - those that paid *per person* more than 13.5 had a higher survival rate.  This is also the proxy of highest class / richest passengers out of 1st class. 
 - `Pclass_3` - class 3 (lowest class) passengers had the lowest survival rate
#### Average priority features - the order is different per specific classifier
 - `Age Bin_42+` - older passengers had a relatively lower rate of survival
 - `Title_Mrs`, `Title_Miss` - women had a higher survival rate
 - `DeckBin_DE` - a mixed deck with different classes, seems that even lower classes had a higher survival rate there.  Perhaps once the deck was saved, everyone there was taken regardless of class.  In other cases, higher class decks were handled first.
 - `DeckBin_unknown_T` - unknown decks had a very low survival rate (perhaps specific area on this ship where few survived)
 - `Age Bin_-4` - youngest passengers had higher survival rate.  High correlation with `Title_Miss`
 - `Age Bin_24-32` - seems that younger adults had a higher survival rates, perhaps they moved faster and were stronger
 - `Pclass_1` - first class had a higher survival rate. Due to another feature of `Fare 13.5+` (part of first class passengers) which got a higher importance, `Pclass_1` got lower importance.  But if `Fare 13.5+` would be removed, `Pclass_1` would get an extremely high importance. Some of those that paid > 13.5 were class 2, and some in class 1 had fare 0
#### Features that made a difference, but not as important - the order is different per specific classifier
 - `Pclass_2` - had a higher survival, higher than class 3 and lower than class 1
 - `Age Bin_11-24` - ages 11-24 had a very slightly higher survival rate
 - `Age Bin_32-42` - ages 32-42 had a slightly lower survival rate than rest
 
 ## Features not used:
 - `Family size` - sum of `` + `` + 1 - didn't help the model, perhaps because of high correlation with `Ticket_Frequency` above  
 - `SibSp` - same
 - `Parch` - same
 - `Sex` - didn't help the model, perhaps because of high correlation with `Title` above  
 - `Embarked` - didn't help the model
  **Many other features were tried that were later removed because didn't help the model or had very high correlation with other features - see git history and `RESULTS.MD`**  
  
## More notes:
- Imputing was done on 'Age' for missing values with `RandomForestRegressor` using other features
- Scaling was done on all features
