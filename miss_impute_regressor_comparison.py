import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNet, RANSACRegressor
from sklearn.svm import SVR
import seaborn as sns
from predict_MC import MC_model
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from matplotlib.colors import LinearSegmentedColormap

n = 1000
X = np.random.randn(n, 2)
X[:,0] = X[:,0] + 2
X[:,1] = X[:,1] + 1
sigma = 1

dataset_names = ['linear', 'square', 'cubic']

ys = []
for i, pattern in enumerate(dataset_names):
    coef = np.random.randn(2, 1)
    ys.append(np.matmul(X**(i+1), coef) + np.random.randn(n,1))

clf = LinearRegression()
clf.fit(X, ys[0])

plt.scatter(ys[0], clf.predict(X))

plt.figure(figsize=(24, 6))

for i in range(3):
    df = pd.DataFrame(np.hstack([X, ys[i]]))
    df.columns = ['x1', 'x2', 'y']
    plt.subplot(1, 3, i+1)
    ax = sns.scatterplot(x="x1", y="x2", hue='y', data=df).set_title(dataset_names[i], fontsize=50)

plt.savefig('image/regressor_simulation_data_scatterplot.png', dpi=200)
plt.show()


names = ['Linear Regression', 'RANSAC', 'SVM', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
regressors = [LinearRegression, RANSACRegressor, SVR, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor]

mse_table = np.zeros([len(names), len(dataset_names)])
# iterate over datasets
for j, y in enumerate(ys):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    for i, name, regressor in zip(range(len(names)), names, regressors):
        clf = regressor()
        clf.fit(X=X_train, y=y_train)
        pred = clf.predict(X_test)
        mse_table[i,j] = mean_squared_error(y_test, pred)
        print((name + ' in ' + dataset_names[j] + ': ' + '{:.3f}'.format(mse_table[i,j])).rjust(50))
        
    print('\n')

mse_table = pd.DataFrame(mse_table, index=names, columns=dataset_names)


datasets = [(X, y) for y in ys]

rng = np.random.RandomState(2)
miss_index = rng.choice(X_test.shape[0], int(n/10))

rmse_median_impute = pd.DataFrame(np.empty([len(names), len(dataset_names)]))
rmse_median_impute.index = names
rmse_median_impute.columns = dataset_names

# iterate over datasets
for j, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    train_median = np.median(X_train[:,0])
    X_test[miss_index,1] = train_median

    for i, name, regressor in zip(range(len(names)), names, regressors):
        clf = regressor()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        rmse_median_impute.iloc[i,j] = np.sqrt(mean_squared_error(y_test, pred))
        print((name + ' in ' + dataset_names[j] + ': ' + '{:.3f}'.format(rmse_median_impute.iloc[i,j])).rjust(50))
        
    print('\n')

    
rmse_mean_impute = pd.DataFrame(np.empty([len(names), len(dataset_names)]))
rmse_mean_impute.index = names
rmse_mean_impute.columns = dataset_names

# iterate over datasets
for j, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    train_mean = np.mean(X_train[:,0])
    X_test[miss_index,1] = train_mean

    for i, name, regressor in zip(range(len(names)), names, regressors):
        clf = regressor()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        rmse_mean_impute.iloc[i,j] = np.sqrt(mean_squared_error(y_test, pred))
        print((name + ' in ' + dataset_names[j] + ': ' + '{:.3f}'.format(rmse_mean_impute.iloc[i,j])).rjust(50))
        
    print('\n')

    
rmse_zero_impute = pd.DataFrame(np.empty([len(names), len(dataset_names)]))
rmse_zero_impute.index = names
rmse_zero_impute.columns = dataset_names

# iterate over datasets
for j, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    train_zero = 0
    X_test[miss_index,1] = train_zero

    for i, name, regressor in zip(range(len(names)), names, regressors):
        clf = regressor()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        rmse_zero_impute.iloc[i,j] = np.sqrt(mean_squared_error(y_test, pred))
        print((name + ' in ' + dataset_names[j] + ': ' + '{:.3f}'.format(rmse_zero_impute.iloc[i,j])).rjust(50))
        
    print('\n')

    
    
    
rmse_MC_impute = pd.DataFrame(np.empty([len(names), len(dataset_names)]))
rmse_MC_impute.index = names
rmse_MC_impute.columns = dataset_names

# iterate over datasets
for j, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)
    X_test[miss_index,1] = np.nan

    for i, name, regressor in zip(range(len(names)), names, regressors):
        clf = regressor()
        clf.fit(X_train, y_train)
        MC_model_obj = MC_model(clf, X_train)
        pred = MC_model_obj.predict_MC(X_test)
        rmse_MC_impute.iloc[i,j] = np.sqrt(mean_squared_error(y_test, pred))
        print((name + ' in ' + dataset_names[j] + ': ' + '{:.3f}'.format(rmse_MC_impute.iloc[i,j])).rjust(50))
        
    print('\n')


rmse_array = np.array([rmse_median_impute.values, rmse_mean_impute.values, 
                       rmse_zero_impute.values, rmse_MC_impute.values])
rmse_array.argmin(axis=0)

rmse_MC_order = pd.DataFrame(1 + rmse_array.argsort(axis=0).argsort(axis=0)[3,:,:])
rmse_MC_order.index = names
rmse_MC_order.columns = dataset_names



sns.set()

myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))

cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

ax = sns.heatmap(rmse_MC_order, cmap=cmap, linewidths=.5, linecolor='lightgray')

# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([1, 2, 3])
colorbar.set_ticklabels(['1', '2', '3'])

# X - Y axis labels
ax.set_ylabel('Regressors')
ax.set_xlabel('Simulation Scheme')

# Only y-axis labels need their rotation set, x-axis labels already have a rotation of 0
_, labels = plt.yticks()
plt.setp(labels, rotation=0)
plt.tight_layout()
plt.savefig('image/Monte_Carlo_Regression_Imputation_rank.png', dpi=200)
plt.show()
