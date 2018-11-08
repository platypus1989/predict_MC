import numpy as np
  
class MC_model:
    def __init__(self, model, train_array, random_state=1):
        self.model = model
        self.train_array = train_array
        self.random_state = random_state
        
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
    
            np.random.seed(self.random_state)
            
            MC_pred_array[:, miss_ind] = temp_train_array[: , miss_ind][np.random.choice(range(temp_train_array.shape[0]), total_sample_n), :]
    
        return MC_pred_array
        
    def predict_proba_MC(self, pred_array, impute_ind = None, sample_n = 100):
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError("{} does not have attribute predict_proba".format(self.model.__class__.__name__))
    
        if impute_ind is None:
            
            impute_ind = range(self.train_array.shape[1])
    
        n = pred_array.shape[0]
    
        class_n = len(self.model.classes_)
    
        pred_proba = np.empty([n, class_n])
    
        for i in range(n):
    
            MC_pred_array = self.MC_miss(pred_array[i, :], impute_ind, sample_n)
    
            pred_proba[i, :] = self.model.predict_proba(MC_pred_array).mean(axis=0)
    
        return pred_proba
        

    def predict_MC(self, pred_array, impute_ind = None, sample_n = 100):
        
        if hasattr(self.model, 'predict_proba'):
            pred_proba = self.predict_proba_MC(pred_array, impute_ind, sample_n)
            return self.model.classes_[pred_proba.argmax(axis=1)]
        else:
            n = pred_array.shape[0]
        
            pred = np.empty([n])
        
            for i in range(n):
        
                MC_pred_array = self.MC_miss(pred_array[i, :], impute_ind, sample_n)
        
                pred[i] = self.model.predict(MC_pred_array).mean()
        
            return pred
            
            
            
        
    
    