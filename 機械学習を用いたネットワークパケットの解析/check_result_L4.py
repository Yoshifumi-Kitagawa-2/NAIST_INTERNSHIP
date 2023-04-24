#!/usr/bin/env python3

import optuna
from optuna import Trial, visualization

study = optuna.load_study(study_name='DDoS_detection_result_L4',
                                storage='sqlite:///DDoS_detection_result_L4.db')

print ('Number of Trails: ', len(study.trials))
# 最適なハイパーパラメータを出力
print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value:.5f}')
print('  Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

Number of Trails:  44924
Best trial:
  Value: 0.94004
  Params:
    batch: 49
    dropout_rate: 0.05458004013794146
    learning_rate: 0.001648506479871434
    n_units_layer0: 76
    n_units_layer1: 75
    n_units_layer2: 88
    n_units_layer3: 39