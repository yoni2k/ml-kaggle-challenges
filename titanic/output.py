import time
import json
import os
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

def get_hard_preds(preds):
    preds_hard = pd.DataFrame({'PassengerId': range(892, 1310), 'Survived': preds.apply(lambda x: int(round(x.mean())), axis=1)})
    return preds_hard


def get_soft_preds(test_probas):
    preds_soft = pd.DataFrame({'PassengerId': range(892, 1310), 'Survived': test_probas.apply(lambda x: 1 - int(round(x.mean())), axis=1)})
    return preds_soft


def get_ensembled_preds(train_probas, y_train, test_probas, max_depth):
    model = ExtraTreesClassifier(max_depth=max_depth, n_estimators=150)
    model.fit(train_probas, y_train)

    train_acc = model.score(train_probas, y_train).round(3)

    cross_3 = cross_val_score(model, train_probas, y_train, cv=3)
    cross_3_acc = cross_3.mean().round(3)
    cross_3_acc_std = cross_3.std().round(3)
    cross_5 = cross_val_score(model, train_probas, y_train, cv=5)
    cross_5_acc = cross_5.mean().round(3)
    cross_5_acc_std = cross_5.std().round(3)
    cross_10 = cross_val_score(model, train_probas, y_train, cv=5)
    cross_10_acc = cross_10.mean().round(3)
    cross_10_acc_std = cross_10.std().round(3)

    print(f'max_depth: {max_depth}, train score: {train_acc}, cross_3_acc: {cross_3_acc}, cross_3_acc_std: {cross_3_acc_std}, cross_3_acc_min_3std: {(cross_3_acc - 3*cross_3_acc_std).round(3)}, cross_5_acc: {cross_5_acc}, cross_5_acc_std: {cross_5_acc_std}, cross_5_acc_min_3std: {(cross_5_acc - 3*cross_5_acc_std).round(3)}, cross_10_acc: {cross_10_acc}, cross_10_acc_std: {cross_10_acc_std}, cross_10_acc_min_3std: {(cross_10_acc - 3*cross_10_acc_std).round(3)}')
    preds_ensemble = model.predict(test_probas)
    return pd.DataFrame({'PassengerId': range(892, 1310), 'Survived': preds_ensemble})

def main():
    preds = pd.DataFrame()
    train_probas = pd.DataFrame()
    test_probas = pd.DataFrame()

    for file_dic in options['files']:
        preds[file_dic['algo']] = pd.read_csv(file_dic['file'] + '/preds.csv')[file_dic['algo']]
        train_probas[file_dic['algo']] = pd.read_csv(file_dic['file'] + '/train_probas.csv')[file_dic['algo'].split('_')[0]]
        test_probas[file_dic['algo']] = pd.read_csv(file_dic['file'] + '/test_probas.csv')[file_dic['algo'].split('_')[0]]

        if file_dic['algo'] == options['best_global_cross_acc']:
            preds_best_cross = pd.DataFrame({'PassengerId': range(892, 1310), 'Survived': preds[file_dic['algo']]})
        if file_dic['algo'] == options['best_global_cross_acc_min_3std']:
            preds_best_cross_min_3std = pd.DataFrame({'PassengerId': range(892, 1310), 'Survived': preds[file_dic['algo']]})

    y_train = pd.read_csv('input/train.csv', index_col='PassengerId')['Survived']

    output_folder = 'output/_' + time.strftime("%Y_%m_%d_%H_%M_%S") + '_outputs/'
    os.mkdir(output_folder)
    with open(output_folder + 'inputs.json', 'w') as json_file:
        json.dump(options, json_file)

    test_probas.corr().to_csv(output_folder + 'probs_correlations.csv')
    preds.corr().to_csv(output_folder + 'preds_correlations.csv')

    preds_best_cross.to_csv(output_folder + 'best_cross.csv', index=False)
    preds_best_cross_min_3std.to_csv(output_folder + 'best_cross_min_3std.csv', index=False)
    get_hard_preds(preds).to_csv(output_folder + 'hard_voting.csv', index=False)
    get_soft_preds(test_probas).to_csv(output_folder + 'soft_voting.csv', index=False)
    get_ensembled_preds(train_probas, y_train, test_probas, 1).to_csv(output_folder + 'ensembled 1.csv', index=False)
    get_ensembled_preds(train_probas, y_train, test_probas, 2).to_csv(output_folder + 'ensembled 2.csv', index=False)
    get_ensembled_preds(train_probas, y_train, test_probas, 3).to_csv(output_folder + 'ensembled 3.csv', index=False)
    get_ensembled_preds(train_probas, y_train, test_probas, 4).to_csv(output_folder + 'ensembled 4.csv', index=False)
    get_ensembled_preds(train_probas, y_train, test_probas, 5).to_csv(output_folder + 'ensembled 5.csv', index=False)
    get_ensembled_preds(train_probas, y_train, test_probas, 6).to_csv(output_folder + 'ensembled 6.csv', index=False)
    get_ensembled_preds(train_probas, y_train, test_probas, 7).to_csv(output_folder + 'ensembled 7.csv', index=False)
    get_ensembled_preds(train_probas, y_train, test_probas, 8).to_csv(output_folder + 'ensembled 8.csv', index=False)
    get_ensembled_preds(train_probas, y_train, test_probas, 9).to_csv(output_folder + 'ensembled 9.csv', index=False)

options = {
    'files': [
        {'file': 'output/best/2019_12_30_11_23_12/Fam_Bin_Fare_Log+13.5_Age_Bin',
         'algo': 'Log_Single'
         },
        # {'file': 'output/best/2019_12_30_11_23_12/Fam_Bin_Fare_Log+13.5_Age_Bin',
        # 'algo': 'XGB_Single'
        # },
        {'file': 'output/best/2019_12_30_11_23_12/Fam_Bin_Fare_Log+13.5_Age_Bin',
          'algo': 'ET 5_Single'
          },
        {'file': 'output/best/2019_12_30_11_23_12/Fam_Bin_Fare_Log+13.5_Age_Num',
         'algo': 'KNN 8_Single'
         },
        # {'file': 'output/best/2019_12_30_11_23_12/Fam_Num_Fare_Log+13.5_Age_Num',
        # 'algo': 'RF 7_Single'
        # }
    ],
    'best_global_cross_acc': 'ET 5_Single',
    'best_global_cross_acc_min_3std': 'Log_Single',
}

main()