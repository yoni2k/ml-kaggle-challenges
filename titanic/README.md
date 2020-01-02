## Solution of titanic dataset from Kaggle
Link to Kaggle: https://www.kaggle.com/c/titanic

**Link to a lot more explanations about how the model works and it's results in**: https://github.com/yoni2k/ml-kaggle-challenges/blob/master/titanic/Simplest%20bottom%20line%20Kernel%20for%20Kaggle.ipynb or on Kaggle https://www.kaggle.com/yoni2k/top-3-with-only-4-features-no-data-leakage

## Main files
- Investigation notebooks - in main directory. Order advised to go through:
    - `initial-titanic-logistic-regression.ipynb`
    - `Fare_investigation.ipynb`
    - `age_investigation.ipynb`
    - `more_feature_engineering.ipynb`
    - `Advanced feature engineering.ipynb`
    - `Same last name different deck, embarked, pclass.ipynb`
- `titanic.py` - main and only code file that does all steps from preprocessing to producing predictions
- `ouptput.py` - takes predictions and probabilites created by `titanic.py`, does some voting / ensembling and creates the actual predictions
- `input` folder:
  - `train.csv` - training input file with expected label 
  - `test.csv` -  test input file
  - `test_answers.csv` - True North.  Not used not to create leakage, left here for completeness
- `output` folder: **outputs of executions per date**
  - `input_options.csv` - inputs for the specific run
  - `results.csv` - results of all classifiers used: 
     - Feature options - various features were created by different options (like binning or not binning a numeric feature) 
     - Train accuracy
     - cross validation accuracy
     - cross validation STD
     - cross validatoin accuracy - 3 * STD - lowest accuracy with 99.7% certainty 
     - Num features - number of features chosen by the model
     - Features - actual features chosen as the important ones by the model
  - **Per specific feature view variation**
    - `classifiers_correlations.csv` - correlations between predictions of different classifiers' used
    - `preds.csv` files - predictions of all models used per view. To create explict submission file, use `output.py`
    - `test_probas.csv` and `train_probas.csv` - probability of test set, to be used for ensembling by `output.py`
    
## Classifiers used
- `LogisticRegression`
- `KNeighborsClassifier`
- `SVC` - classifier based on Support Vector Machines
- `GaussianNB` - Naive Bayes
- `RandomForestClassifier`
- `ExtraTreesClassifier`
- `XGBClassifier` from xgboost library
** For most classifiers, Grid Search was done to find the best hyperparameters**
  
## Ensembling done:
- Soft voting
- Hard voting
- `ExtraTreesClassifier` based on `predict_proba_`

## Results:
  - ExtraTreesClassifier gets 84% locally with cross validation, and 82% on Kaggle
  - Doing Ensembling with `LogisticRegression` and `KNeighborsClassifier` gets 82.2% on Kaggle. 
  See more details on: 
    https://github.com/yoni2k/ml-kaggle-challenges/blob/master/titanic/Simplest%20bottom%20line%20Kernel%20for%20Kaggle.ipynb or on Kaggle https://www.kaggle.com/yoni2k/top-3-with-only-4-features-no-data-leakage

## Some of the main Features used:
  - `Title` - retrieved from `Name` feature, and grouped into cagetories, then added dummies
  - `Deck` - retrieved from first letter of `Cabin` feature, and grouped into cagetories, then added dummies. 
    Only categories that helped the model were left.
  - `Pclass`
  - `Fare` per person of > 13.5 (needed to divide `Fare` by `Ticket_Frequency`). Other categories and splits, using as is, or doing `Log` on the numbers didn't help the model
  - `Age` - binned by survival rate.  See about imputing below
  - `Known family/ticket survived %` - look both based on family (last name) and ticket, and add average % of survived in training set.
 
 ## Features that explain the most the survival rate:
 *Notes:* 
 - *Mostly based on Logistic Regression and Randon Forest feature importance info*
 - *See notebooks in this folder for much better BI of connections between survival and different features*  
  
## More notes:
- Imputing was done on 'Age' for missing values with `RandomForestRegressor` using other features
- Scaling was done on all features with `MinMaxScaler`
- Feature selection is done per fold of cross validation
- Scaling is done per fold of cross validation
- Cross validation is done using full flow of feature selection, scaling etc.

# OLD discussion about Feature importance
**Mostly relevant**, but new version allows each model to choose best features from maximum available based on `RFECV`, so different models reach different conclusions.

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
