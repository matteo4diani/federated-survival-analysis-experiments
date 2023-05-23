import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
import matplotlib
matplotlib.use("QtAgg")
from matplotlib import pyplot as plt
import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

def load_data():
    df_train = metabric.read_df()
    df_test = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_test.index)
    df_val = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_val.index)
    
    return preprocess_data(df_train, df_val, df_test)
    
    
def preprocess_data(df_train, df_val, df_test):
    cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
    cols_leave = ['x4', 'x5', 'x6', 'x7']

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(standardize + leave)
    
    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')
    
    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    durations_test, events_test = get_target(df_test)
    
    train = x_train, y_train
    val = x_val, y_val
    test = x_test, durations_test, events_test
    
    return train, val, test 
    
def build_net(x_train, device):
    in_features = x_train.shape[1]
    num_nodes = [128, 64, 64, 64]
    out_features = 1
    batch_norm = True
    dropout = 0.2
    output_bias = False

    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                dropout, output_bias=output_bias)


    model = CoxPH(net, tt.optim.Adam, device=device)
    model.optimizer.set_lr(0.001)
    return model

    
def train(net, train, val, epochs=512, device=None, batch_size=256):
    callbacks = [tt.callbacks.EarlyStopping(patience=10)]
    verbose = True
    
    return net.fit(train[0],
                   train[1],
                   epochs=epochs,
                   batch_size=batch_size,
                   verbose=verbose, 
                   val_data=val,
                   callbacks=callbacks,
                   val_batch_size=batch_size)
    
def test(net, test):
    _ = net.compute_baseline_hazards()
    surv = net.predict_surv_df(test[0])
    durations_test, events_test = test[1], test[2]
    
    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')

    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    return ev, time_grid    

def main():
    np.random.seed(1234)
    _ = torch.manual_seed(123)
    DEVICE = "cuda" if torch.cuda.is_available()  else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("Centralized PyTorch training")
    print("Load data")
    
    train_data, val_data, test_data = load_data()
    
    print("Start training")

    net = build_net(train_data[0], device=DEVICE)
    log = train(net=net, epochs=1000000, train=train_data, val=val_data, device=DEVICE)
    #_ = log.plot()
    #plt.show(block=False)
    input("Press Enter to continue...")
    print(f"Loss: {log.get_measures()}")
   
    print("Evaluate model")
    
    ev, time_grid = test(net=net, test=test_data)
    #_ = ev.brier_score(time_grid).plot()
    #plt.show(block=False)
    input("Press Enter to continue...")
    
    print(f"Integrated Brier Score: {ev.integrated_brier_score(time_grid)}")
        
if __name__ == "__main__":
    main()
