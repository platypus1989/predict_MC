import numpy as np
from time import time


def MC_miss(train_array, pred_obs, impute_ind, sample_n = 100):

    miss_ind = np.intersect1d(np.where(np.isnan(pred_obs))[0], impute_ind)

    temp_pred_array = pred_obs.reshape([1, len(pred_obs)])

    if len(miss_ind) == 0:

        MC_pred_array = temp_pred_array

    else:

        total_sample_n = sample_n # pow(sample_n, len(miss_ind))

        MC_pred_array = np.repeat(temp_pred_array, total_sample_n, axis=0)

        temp_train_array = train_array[~ (train_array[:, miss_ind] == 0).max(axis=1), :]

        MC_pred_array[:, miss_ind] = temp_train_array[: , miss_ind][np.random.choice(range(temp_train_array.shape[0]), total_sample_n), :]

    return MC_pred_array


def predict_proba_MC(model, train_array, pred_array, impute_ind = None, sample_n = 100):
    
    if impute_ind is None:
        
        impute_ind = range(train_array.shape[1])

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
    

def predict_MC(model, train_array, pred_array, impute_ind = None, sample_n = 100):
    
    pred_proba = predict_proba_MC(model, train_array, pred_array, impute_ind, sample_n)
    
    return model.classes_[pred_proba.argmax(axis=1)]
    
    
class MC_model:
    def __init__(self, model, train_array):
        self.model = model
        self.train_array = train_array
        
    def MC_miss(self, pred_obs, impute_ind = None, sample_n = 100):
        
        if impute_ind is None:
            impute_ind = range(self.train_array.shape[1])

        miss_ind = np.intersect1d(np.where(np.isnan(pred_obs))[0], impute_ind)
    
        temp_pred_array = pred_obs.reshape([1, len(pred_obs)])
    
        if len(miss_ind) == 0:
    
            MC_pred_array = temp_pred_array
    
        else:
    
            total_sample_n = sample_n # pow(sample_n, len(miss_ind))
    
            MC_pred_array = np.repeat(temp_pred_array, total_sample_n, axis=0)
    
            temp_train_array = self.train_array[~ (self.train_array[:, miss_ind] == 0).max(axis=1), :]
    
            MC_pred_array[:, miss_ind] = temp_train_array[: , miss_ind][np.random.choice(range(temp_train_array.shape[0]), total_sample_n), :]
    
        return MC_pred_array
        
    def predict_proba_MC(self, pred_array, impute_ind = None, sample_n = 100):
    
        if impute_ind is None:
            
            impute_ind = range(self.train_array.shape[1])
    
        n = pred_array.shape[0]
    
        class_n = len(self.model.classes_)
    
        pred_proba = np.empty([n, class_n])
    
        tic = time()
    
        for i in range(n):
    
            MC_pred_array = self.MC_miss(pred_array[i, :], impute_ind, sample_n)
    
            pred_proba[i, :] = self.model.predict_proba(MC_pred_array).mean(axis=0)
    
            if i == 100:
    
                print('it will take approximately ', (time() - tic)/100*n, ' seconds to finish...')
    
        return pred_proba
        

    def predict_MC(self, pred_array, impute_ind = None, sample_n = 100):
        
        pred_proba = self.predict_proba_MC(pred_array, impute_ind, sample_n)
        
        return self.model.classes_[pred_proba.argmax(axis=1)]
    
    