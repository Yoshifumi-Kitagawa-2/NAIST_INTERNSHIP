#!/usr/bin/env python3

import optuna
from optuna import Trial, visualization

study = optuna.load_study(study_name='DDoS_detection_result_L5',
                                storage='sqlite:///DDoS_detection_result_L5.db')

print ('Number of Trails: ', len(study.trials))
# 最適なハイパーパラメータを出力
print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value:.5f}')
print('  Params:')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

Number of Trails:  39909
Best trial:
  Value: 0.93923
  Params:
    batch: 62
    dropout_rate: 0.12879005505329943
    learning_rate: 0.0017433095513950604
    n_units_layer0: 45
    n_units_layer1: 91
    n_units_layer2: 23
    n_units_layer3: 27
    n_units_layer4: 46