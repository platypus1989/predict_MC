import numpy as np
import pandas as pd
from time import time

def MC_miss(train_array, pred_obs, impute_ind = range(6), sample_n = 100):

    miss_ind = np.intersect1d(np.where(pred_obs == 0)[0], impute_ind)

    temp_pred_array = pred_obs.reshape([1, len(pred_obs)])

   

    if len(miss_ind) == 0:

        MC_pred_array = temp_pred_array

    else:

        total_sample_n = sample_n # pow(sample_n, len(miss_ind))

        MC_pred_array = np.repeat(temp_pred_array, total_sample_n, axis=0)

        temp_train_array = train_array[~ (train_array[:, miss_ind] == 0).max(axis=1), :]

        MC_pred_array[:, miss_ind] = temp_train_array[: , miss_ind][np.random.choice(range(temp_train_array.shape[0]), total_sample_n), :]

       

    return MC_pred_array


def predict_proba_MC(model, train_array, pred_array, impute_ind = range(6), sample_n = 100):

    n = pred_array.shape[0]

    class_n = len(model.classes_)

    pred_proba = np.empty([n, class_n])

   

    tic = time()

    for i in range(n):

        MC_pred_array = MC_miss(train_array, pred_array[i, :], impute_ind, sample_n)

        pred_proba[i, :] = model.predict_proba(MC_pred_array).mean(axis=0)

        if i == 100:

            print('it will take approximately ', (time() - tic)/100*n, ' seconds to finish...')

  

    return pred_proba