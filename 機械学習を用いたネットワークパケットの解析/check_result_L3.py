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