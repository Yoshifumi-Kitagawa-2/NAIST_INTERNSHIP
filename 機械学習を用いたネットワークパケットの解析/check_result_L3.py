#!/usr/bin/env python3

import optuna
from optuna import Trial, visualization

study = optuna.load_study(study_name='DDoS_detection_result_L3',
                                storage='sqlite:///DDoS_detection_result_L3.db')

print ('Number of Trails: ', len(study.trials))
# 最適なハイパーパラメータを出力
print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value:.5f}')
print('  Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

    Number of Trails:  43865
Best trial:
  Value: 0.94055
  Params:
    batch: 57
    dropout_rate: 0.10040026456923512
    learning_rate: 0.0018596447776440274
    n_units_layer0: 46
    n_units_layer1: 70
    n_units_layer2: 50