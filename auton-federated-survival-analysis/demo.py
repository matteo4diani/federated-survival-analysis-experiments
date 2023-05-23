from matplotlib import pyplot as plt
from auton_survival import datasets, preprocessing, metrics
from auton_survival.models.cph import DeepCoxPH, DeepRecurrentCoxPH
import numpy as np
import pandas as pd
import torch
from torch import unsqueeze
# Load the SUPPORT Dataset
outcomes, features = datasets.load_dataset("SUPPORT")

cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 
             'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 
             'glucose', 'bun', 'urine', 'adlp', 'adls']

# Preprocess (Impute and Scale) the features
features = preprocessing.Preprocessor().fit_transform(features, cat_feats, num_feats)

# Train a Deep Cox Proportional Hazards (DCPH) model
model = DeepRecurrentCoxPH()
print(outcomes.time)
print(outcomes.event)

model.fit(features, torch.tensor(outcomes.time.values), torch.tensor(outcomes.event.values))

# Predict risk at specific time horizons.
times = list(np.linspace(30, 2000, num=100))
predictions = model.predict_risk(features, t=times)

ibs = metrics.survival_regression_metric('ibs', outcomes, predictions, times)
print(f'IBS: {ibs}')
input("Forward...?")