# Open env as: conda activate /data/smangalik/myenvs/rdds
# Run as: python discontinuity_prediction.py
import os

import scipy.stats as stats
import pandas as pd
import numpy as np
from scipy.stats import linregress
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import optuna
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import ShuffleSplit, train_test_split
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set CUDA_VISIBLE_DEVICES to -1 to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

global best_dev_loss_ever
global best_params_ever
best_dev_loss_ever = float('inf')
best_params_ever = {}

score = "Anxiety"
score_cov = "Depression"

# Assign a location for storing artifacts
artifact_base_path = "/data/smangalik/rdd_models"
artifact_store = optuna.artifacts.FileSystemArtifactStore(base_path=artifact_base_path)

first_case = "/users2/smangalik/causal_modeling/rdd_first_covid_case_FREEZE.csv"
df = pd.read_csv(first_case)
df['fips'] = df['fips'].astype(str).str.zfill(5)
df_cov = df[df['feat']==score_cov]
df = df[df['feat']==score]

print("\nRDD Dataframe, row count:", len(df))
print(df.head())
print("Columns:",df.columns)

print("\nRDD Dataframe with covariates, row count:", len(df_cov))

# Load county embeddings
#county_embeddings_csv = "/users2/smangalik/causal_modeling/ctlb2.feat$roberta_la_meL23nosent_wavg$ctlb_2020$county.csv"
#county_embeddings_csv = "/users2/smangalik/causal_modeling/ctlb2.feat$roberta_la_meL23dlatk_avg$ctlb_2020$county.csv"
#county_embeddings_csv = "/users2/smangalik/causal_modeling/ctlb2.feat$roberta_la_meL23nosent_avg$ctlb_2020$county.csv"
# county_embeddings_csv = "/data/smangalik/county_embeddings/ctlb2.feat$roberta_la_meL23nosent_wavg$ctlb_2020$full_county_set.csv"
county_embeddings_csv = "/data/smangalik/county_embeddings/final_county_wavg_emb_wide.csv"
county_embeddings = pd.read_csv(county_embeddings_csv)
county_embeddings = county_embeddings.rename(columns={'group_id':'fips'})
county_embeddings['fips'] = county_embeddings['fips'].astype(str).str.zfill(5)
embeddings_cols = [col for col in county_embeddings.columns if 'me' in col]
print("Loaded", len(county_embeddings), "county embeddings from", county_embeddings_csv)
print(county_embeddings)

# mean of all the embedding columns in the dataframe
df_mean = county_embeddings[embeddings_cols].mean()
#mean_embedding = [df_mean[col] for col in embeddings_cols]
#print("\nMean embedding length:", len(mean_embedding))

# Join with the dataframe
df = df.merge(county_embeddings, on='fips', how='left')
print("\nRDD Dataframe with embeddings, row count:", len(df))
print(df.head())

# Number of rows with NaN values in the embeddings
nan_rows = df[df[embeddings_cols].isna().any(axis=1)]
print("\nNumber of rows with NaN values in the embeddings:", len(nan_rows))

def regress_missing_points(dat_x, dat_y):
    non_nan_indices = [i for i, value in zip(dat_x,dat_y) if not np.isnan(value)]
    non_nan_values = [value for value in dat_y if not np.isnan(value)]
    if len(non_nan_indices) < 2: # Not enough data points to perform regression
        return [np.nan], np.nan, np.nan
    slope, intercept, _, _, _ = linregress(non_nan_indices, non_nan_values)
    imputed_data = [value if not np.isnan(value) else slope * i + intercept for i, value in zip(dat_x,dat_y)]
    return imputed_data, slope, intercept

def generate_training_data(df:pd.DataFrame, df_cov:pd.DataFrame, buffer:int=0):
    
    assert(len(df) == len(df_cov)), "RDD and covariate dataframes must have the same length"
    
    # Generate training data for the model
    print("\nProcessing training data with a buffer size of",buffer)
    X_counties = []
    X = []
    X_cov  = []
    X_cov_line_params  = []
    X_line_params = []
    X_ses = []
    X_embedding = []
    y = []
    for i, row in tqdm(df.iterrows()): # Iterate over the counties
        
        row_cov = df_cov.iloc[i]

        # Parse list from string
        target_before = row['target_before'].replace('[','').replace(']','').split() 
        target_after = row['target_after'].replace('[','').replace(']','').split()
        cov_before = row_cov['target_before'].replace('[','').replace(']','').split()
        cov_after = row_cov['target_after'].replace('[','').replace(']','').split()
        # Convert to float
        target_before = [float(t) for t in target_before] 
        target_after = [float(t) for t in target_after]
        cov_before = [float(t) for t in cov_before]
        cov_after = [float(t) for t in cov_after]
        
        # Load the x-axis
        x_before = np.arange(len(target_before)) - len(target_before) + 1  
        x_after = np.arange(len(target_after))            
        # Regress missing points
        if buffer > 0:
            imputed_before, slope_before, intercept_before = regress_missing_points(x_before[:-buffer], target_before[:-buffer])
            imputed_after, slope_after, intercept_after = regress_missing_points(x_after[buffer:], target_after[buffer:])   
            imputed_cov_before, slope_cov_before, intercept_cov_before = regress_missing_points(x_before[:-buffer], cov_before[:-buffer])  
            imputed_cov_after, slope_cov_after, intercept_cov_after = regress_missing_points(x_after[buffer:], cov_after[buffer:])
        else: # Remove the last point (during the event)
            imputed_before, slope_before, intercept_before = regress_missing_points(x_before[:-1], target_before[:-1])
            imputed_after, slope_after, intercept_after = regress_missing_points(x_after[:-1], target_after[:-1])     
            imputed_cov_before, slope_cov_before, intercept_cov_before = regress_missing_points(x_before[:-1], cov_before[:-1])
            imputed_cov_after, slope_cov_after, intercept_cov_after = regress_missing_points(x_after[:-1], cov_after[:-1])
            
        # If there are any NaN values in the imputed data, skip this row
        if np.isnan(imputed_before).any() or np.isnan([slope_before,intercept_before,slope_after,intercept_after]).any():
            print("\n(!) Skipping", row['county'], "row due to NaN values")
            continue
        
        # Find the first and last points of the regression lines
        regression_before_last =  intercept_before + (slope_before * -buffer)
        regression_after_first = intercept_after + (slope_after * buffer) 
        #regression_cov_before_last = intercept_cov_before + (slope_cov_before * -buffer)
        #regression_cov_after_first = intercept_cov_after + (slope_cov_after * buffer)
        
        # print("\nRegression before last:", regression_before_last)
        # print("Regression after first:", regression_after_first)
        
        # Store counties in order
        X_counties.append(row['county'])
        
        # Data points before 
        X_i = imputed_before
        X.append(X_i)
        X_cov_i = imputed_cov_before # Remove the last point (target_during)
        X_cov.append(X_cov_i)
        
        # The slope and intercept of the regression line before
        X_line_params_i = [slope_before, intercept_before]
        X_line_params.append(X_line_params_i)
        X_cov_line_params_i = [slope_cov_before, intercept_cov_before]
        X_cov_line_params.append(X_cov_line_params_i)
        
        # The socioecconomic status of the county
        X_ses_i = [row['ses'],row['ses3'],row['pblack'],row['pfem'],row['phisp'],row['p65old'],row['unemployment_rate_2018']]
        X_ses.append(X_ses_i)
        
        # The embedding of the county
        X_embedding_i = [row[col] for col in embeddings_cols]
        # If there are any NaN values in the embedding, assign the mean embedding
        #if np.isnan(X_embedding_i).any(): X_embedding_i = mean_embedding
        X_embedding.append(X_embedding_i)
        
        # print("\nX_i:", X_i)
        # print("X_line_params_i:", X_line_params_i)
        
        # Outcomes
        slope_change = slope_after - slope_before
        #intercept_change = intercept_after - intercept_before
        #point_discontinuity = target_after[1] - target_before[-1]
        regression_discontinuity = regression_after_first - regression_before_last
        during_score = np.array(target_after)[~np.isnan(target_after)][0]
        next_score = np.array(target_after)[~np.isnan(target_after)][1]
        y_i = {
            "county":row['county'],
            "Delta b":regression_discontinuity, 
            "{} at t=0".format(score):during_score,
            "{} at t=1".format(score):next_score,
            "Delta m":slope_change,
            "m (after)":slope_after, 
            "b (after)":intercept_after,
        }
        # print("y_i:", y_i)        
        y.append(y_i)
        

        
    X = np.array(X)
    X_line_params = np.array(X_line_params)
    X_ses = np.array(X_ses)
    X_embedding = np.array(X_embedding)
    y = pd.DataFrame(y)
    
    return X, X_counties, X_cov, X_line_params, X_cov_line_params, X_ses, X_embedding, y

X, X_counties, X_cov, X_line_params, X_cov_line_params, X_ses, X_embedding, y = generate_training_data(df, df_cov, buffer=0)

# Turn into a df then run .describe() to get the mean and std of each column
print("\nStats for X LP:")
stats_X_lp = pd.DataFrame(X_line_params, columns=['delta b','delta m'])
print(stats_X_lp.describe())
print("\nStats for y:")
print(y.describe())

# print("\nX:", X.shape)    
# print("X_line_params:", X_line_params.shape)
# print("X_ses:", X_ses.shape)
# print("X_embedding:", X_embedding.shape)
# print("y:", y.shape)  

# Show rows in X_embeddings that contain NaN values
print("County embeddings:", X_embedding.shape)
print(X_embedding)

X_with_line_params = np.concatenate((X, X_line_params), axis=1)
print("\nX_with_line_params:", X_with_line_params.shape)

X_cov_line_params = np.array(X_cov_line_params)
print("\nX_cov_line_params:", X_cov_line_params.shape) 
X_with_cov = np.concatenate((X, X_cov), axis=1)
print("\nX_with_cov:", X_with_cov.shape)
X_with_line_params_and_cov = np.concatenate((X, X_line_params, X_cov), axis=1)
print("\nX_with_line_params_and_cov:", X_with_line_params_and_cov.shape)
X_with_line_params_and_cov_and_cov_line_params = np.concatenate((X, X_line_params, X_cov, X_cov_line_params), axis=1)
print("\nX_with_line_params_and_cov_and_cov_line_params:", X_with_line_params_and_cov_and_cov_line_params.shape)


X_with_line_params_and_ses = np.concatenate((X, X_line_params, X_ses), axis=1)
print("\nX_with_line_params_and_ses:", X_with_line_params_and_ses.shape)

X_with_ses = np.concatenate((X, X_ses), axis=1)
print("\nX_with_ses:", X_with_ses.shape)

X_line_params_with_embedding = np.concatenate((X_line_params, X_embedding), axis=1) 
print("\nX_line_params_with_embedding:", X_line_params_with_embedding.shape)

X_with_embedding = np.concatenate((X, X_embedding), axis=1)
print("\nX_with_embedding:", X_with_embedding.shape)

X_with_line_params_and_embedding = np.concatenate((X, X_line_params, X_embedding), axis=1)
print("\nX_with_line_params_and_embedding:", X_with_line_params_and_embedding.shape)

X_everything = np.concatenate((X, X_line_params, X_cov, X_cov_line_params, X_embedding), axis=1)
print("\nX_everything:", X_everything.shape)

print("\ny:", y.shape)

def evaluate(X, y_df, outcomes=['Delta b','Delta m'], model_name='RidgeRegression', run_name=""):
    print("\nEvaluating", run_name)
    
    y_df = y_df[outcomes].to_numpy()
        
    # Create a stratified 60% train, 20% validation, 20% test split
    sss = ShuffleSplit(n_splits=1, test_size=0.20, random_state=25)
    for train_mask, test_mask in sss.split(X, y_df):
        pass
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y_df[train_mask], y_df[test_mask]
    
    sss = ShuffleSplit(n_splits=1, test_size=0.25, random_state=25)
    for tr_mask, val_mask in sss.split(X_train, y_train):
        pass
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    if model_name == 'RidgeRegression':
        param_grid = {'alpha': [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]}
        model = Ridge(random_state=25)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train, y_train)
        best_alpha = grid_search.best_params_['alpha']
        print("\t\tUsing best alpha from GridSearchCV: {}".format(best_alpha))
        y_pred = grid_search.predict(X_test)
        
    elif model_name == 'KNeighborsRegressor':
        param_grid = {'n_neighbors': [1, 2, 5], 'weights': ['distance'], 'metric': ['euclidean']}
        model = KNeighborsRegressor()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train, y_train)
        print("\t\tUsing best n_neighbors from GridSearchCV: {}".format(grid_search.best_params_['n_neighbors']))
        print("\t\tUsing best weights from GridSearchCV: {}".format(grid_search.best_params_['weights']))
        print("\t\tUsing best metric from GridSearchCV: {}".format(grid_search.best_params_['metric']))
        y_pred = grid_search.predict(X_test)
        
    elif model_name == "NoChange":
        
        y_pred = np.zeros_like(y_test)
        
        for i, y_col in enumerate(outcomes):
        
            train_line_params = X_line_params[train_mask]
            slope_befores = train_line_params[:, 0]
            intercept_befores = train_line_params[:, 1]
            if y_col == "Delta b":  # Guess 0
                y_pred[:, i] = np.zeros(y_test.shape[0])
            elif y_col == "{} at t=0".format(score): # Guess intercept_before
                y_pred[:, i] = intercept_befores
            elif y_col == "{} at t=1".format(score): # Guess slope_before + intercept_before
                y_pred[:, i] = slope_befores + intercept_befores
            elif y_col == "Delta m": # Guess 0
                y_pred[:, i] = np.zeros(y_test.shape[0])
            elif y_col == "m (after)": # Guess slope_before
                y_pred[:, i] = slope_befores
            elif y_col == "b (after)": # Guess intercept_before
                y_pred[:, i] = intercept_befores
            else:
                print("Unknown column:", y_col)
                y_pred[:, i] = np.zeros(y_test.shape[0])
        
    elif model_name == 'MeanBaseline':
        # Always guess the mean value in the training set
        y_train_mean = np.mean(y_train, axis=0)
        y_train_mean = y_train_mean.reshape(1, -1)  # Reshape to match the shape of y_test
        y_pred = np.tile(y_train_mean, (X_test.shape[0], 1))
        
    elif model_name == 'RandomForest':
        if input_dim < 100: # If the input dimension is small, use ExtraTreesRegressor
            param_grid = {'n_estimators': [100, 500], 'max_depth': [10, None]}
        else: # If the input dimension is large, use ExtraTreesRegressor with a larger max_depth
            param_grid = {'n_estimators': [500, 1000], 'max_depth': [None]}
        model = RandomForestRegressor(random_state=25)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=2)
        grid_search.fit(X_train, y_train)
        best_n_estimators = grid_search.best_params_['n_estimators']
        best_max_depth = grid_search.best_params_['max_depth']
        print("\t\tUsing best n_estimators from GridSearchCV: {}".format(best_n_estimators))
        print("\t\tUsing best max_depth from GridSearchCV: {}".format(best_max_depth))
        y_pred = grid_search.predict(X_test)
        
    elif model_name == 'ExtraTrees':
        if input_dim < 100: # If the input dimension is small, use ExtraTreesRegressor
            param_grid = {'n_estimators': [100, 500], 'max_depth': [10, None]}
        else: # If the input dimension is large, use ExtraTreesRegressor with a larger max_depth
            param_grid = {'n_estimators': [500, 1000], 'max_depth': [None]}
        model = ExtraTreesRegressor(random_state=25)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=2)
        grid_search.fit(X_train, y_train)
        best_n_estimators = grid_search.best_params_['n_estimators']
        best_max_depth = grid_search.best_params_['max_depth']
        print("\t\tUsing best n_estimators from GridSearchCV: {}".format(best_n_estimators))
        print("\t\tUsing best max_depth from GridSearchCV: {}".format(best_max_depth))
        y_pred = grid_search.predict(X_test)
        
    elif model_name == 'XGBoost':
        if input_dim < 100: # If the input dimension is small, use ExtraTreesRegressor
            param_grid = {'n_estimators': [100, 500], 'max_depth': [10, None]}
        else: # If the input dimension is large, use ExtraTreesRegressor with a larger max_depth
            param_grid = {'n_estimators': [500, 1000], 'max_depth': [None]}
        model = xgb.XGBRegressor(random_state=25)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=2)
        grid_search.fit(X_train, y_train)
        best_n_estimators = grid_search.best_params_['n_estimators']
        best_max_depth = grid_search.best_params_['max_depth']
        print("\t\tUsing best n_estimators from GridSearchCV: {}".format(best_n_estimators))
        print("\t\tUsing best max_depth from GridSearchCV: {}".format(best_max_depth))
        y_pred = grid_search.predict(X_test)
        
    elif model_name == 'MLPRegressor':
        param_grid = {'alpha': [0.001, 0.005, 0.01, 0.05], 'hidden_layer_sizes': [(100,100),(2,2),(2,2,2,2),(5,5),(5,5,5,5),(11,11),(11,11,11,11)]}
        model = MLPRegressor(max_iter=5000, random_state=25)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train, y_train)
        best_alpha = grid_search.best_params_['alpha']
        best_hidden_layer_sizes = grid_search.best_params_['hidden_layer_sizes']
        print("\t\tUsing best alpha from GridSearchCV: {}".format(best_alpha))
        print("\t\tUsing best hidden_layer_sizes from GridSearchCV: {}".format(best_hidden_layer_sizes))
        y_pred = grid_search.predict(X_test)
        
    elif 'FFN' in model_name:
        
        #  Default Hyperparameters
        epochs = 200 
        batch_size = 64 # len(X_train) // 2
        optimizer_name = 'Adam'
        
        # Tunable Hyperparameters
        lr = 0.01
        layer_size = 100
        num_layers = 2 
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        X_tr_tensor, X_dev_tensor, y_tr_tensor, y_dev_tensor = train_test_split(
                X_train_tensor, y_train_tensor, test_size=0.25, random_state=25)
        
        trainloader = DataLoader(TensorDataset(X_tr_tensor, y_tr_tensor), batch_size=batch_size, shuffle=True)
        devloader = DataLoader(TensorDataset(X_dev_tensor, y_dev_tensor), batch_size=batch_size, shuffle=False)
        
        class TorchRegressor(nn.Module):
            def __init__(self, input_dim, output_dim, 
                            layer_size=100, num_layers=1,
                            lr=0.01, epochs=100, batch_size=64,
                            optimizer_name='Adam'):
                super(TorchRegressor, self).__init__()
                self.lr = lr
                self.epochs = epochs
                self.batch_size = batch_size
                self.optimizer_name = optimizer_name
                layers = []
                layers.append(nn.Linear(input_dim, layer_size))
                layers.append(nn.ReLU())
                for _ in range(num_layers - 1):
                    layers.append(nn.Linear(layer_size, layer_size))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(layer_size, output_dim))
                self.fc = nn.Sequential(*layers)
                # Final layer to output the desired dimension
            def forward(self, x):
                return self.fc(x)
            def predict(self, x):
                with torch.no_grad():
                    return self.fc(x).detach().numpy()
            def fit(self, x, y):
                dataloader = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True) 
                
                optimizer = getattr(optim, self.optimizer_name)(self.parameters(), lr=self.lr)
                criterion = nn.MSELoss()

                for epoch in range(self.epochs):
                    self.train()
                    epoch_loss = 0.0
                    for batch_inputs, batch_targets in dataloader:
                        optimizer.zero_grad()
                        outputs = self(batch_inputs)
                        outputs = outputs.view(batch_targets.shape)
                        loss = criterion(outputs, batch_targets)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
        
        # Reset seed for reproducibility
        torch.manual_seed(25)
        np.random.seed(25)
        
        global best_dev_loss_ever
        global best_params_ever
        best_dev_loss_ever = float('inf')
        best_params_ever = {}

        def optuna_objective(trial):
            if input_dim < 100:
                lr = trial.suggest_categorical('lr', [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
                layer_size = trial.suggest_categorical('layer_size', [2, 5, 10])
                num_layers = trial.suggest_categorical('num_layers', [1])
                epochs = trial.suggest_int('epochs', 50, 300, step=50)
                #batch_size = trial.suggest_int('batch_size', 1, len(X_train)//2)
            else:
                lr = trial.suggest_categorical('lr', [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
                layer_size = trial.suggest_int('layer_size', 200, 1000, step=200)
                num_layers = trial.suggest_categorical('num_layers', [1])
                epochs = trial.suggest_int('epochs', 50, 200, step=50)
                #batch_size = trial.suggest_int('batch_size', 1, len(X_train)//2)
            
            
            model = TorchRegressor(
                input_dim=input_dim, output_dim=output_dim, 
                layer_size=layer_size, num_layers=num_layers,
                lr=lr, epochs=epochs, batch_size=batch_size, 
                optimizer_name=optimizer_name
            )
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            best_dev_loss = float('inf')
            
            # Training Loop
            for epoch in range(epochs): 
                model.train()
                epoch_loss = 0.0
                for batch_inputs, batch_targets in trainloader:
                    optimizer.zero_grad()
                    outputs = model(batch_inputs)
                    outputs = outputs.view(batch_targets.shape)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                # Evaluate on dev set
                model.eval()
                dev_loss = 0.0
                with torch.no_grad():
                    for batch_inputs, batch_targets in devloader:
                        outputs = model(batch_inputs)
                        outputs = outputs.view(batch_targets.shape)
                        loss = criterion(outputs, batch_targets)
                        dev_loss += loss.item()
                        
                    dev_loss = dev_loss / len(devloader)
                    
                    best_dev_loss = min(best_dev_loss, dev_loss)
                    
                    global best_dev_loss_ever
                    if best_dev_loss < best_dev_loss_ever:
                        best_dev_loss_ever = best_dev_loss
                        #print("Best dev loss:", best_dev_loss)
                        torch.save(model.state_dict(), '/data/smangalik/best_ffn.pth')
                        global best_params_ever
                        best_params_ever = {
                            'layer_size': layer_size,
                            'lr': lr,
                            'num_layers': num_layers,
                            'epochs': epochs
                        }
                        
                    trial.report(dev_loss, step=epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
            return best_dev_loss  # Optuna minimizes this loss
        
        sampler = optuna.samplers.TPESampler(seed=25)
        study = optuna.create_study(
            study_name="FFN_Optuna",
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(max_resource=epochs),
        )
        study.optimize(optuna_objective, n_trials=20, show_progress_bar=True)

        # Show best hyperparameters
        # print("Best hyperparameters:", study.best_params, "Best loss:", study.best_value)
        # lr = study.best_params['lr']
        # layer_size = study.best_params['layer_size']
        # num_layers = study.best_params['num_layers']
        # epochs = study.best_params['epochs']
        # optimizer_name = study.best_params['optimizer']
        # epochs = study.best_params['epochs']
        # batch_size = study.best_params['batch_size']
        
        print("Best hyperparameters:", best_params_ever, "Best loss:", best_dev_loss_ever)
        lr = best_params_ever['lr']
        layer_size = best_params_ever['layer_size']
        num_layers = best_params_ever['num_layers']
        epochs = best_params_ever['epochs']
                
        # Run with best hyperparameters from Optuna
        model = TorchRegressor(
            input_dim=input_dim, output_dim=output_dim, 
            lr=lr, epochs=epochs, 
            layer_size=layer_size, num_layers=num_layers,
            batch_size=batch_size,
            optimizer_name=optimizer_name)
        
        model.load_state_dict(torch.load('/data/smangalik/best_ffn.pth'))
        #model.fit(X_tr_tensor, y_tr_tensor)
        
        y_pred = model.predict(X_test_tensor)
    
    elif model_name == 'GRU':
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        # Default Hyperparameters
        epochs = 200 
        batch_size = 64
        
        # Tunable Hyperparameters
        learning_rate = 0.01
        dropout = 0.1
        layer_size = 100
        num_layers = 2 
        
        # Create a train and dev split
        X_tr_tensor, X_dev_tensor, y_tr_tensor, y_dev_tensor = train_test_split(
                X_train_tensor, y_train_tensor, test_size=0.25, random_state=25)
        
        trainloader = DataLoader(TensorDataset(X_tr_tensor, y_tr_tensor), batch_size=batch_size, shuffle=True)
        devloader = DataLoader(TensorDataset(X_dev_tensor, y_dev_tensor), batch_size=batch_size, shuffle=False)

        class GRURegressor(nn.Module):
            def __init__(self, input_dim, output_dim, 
                        layer_size=100, num_layers=1, 
                        dropout=0.1, batch_size=64, 
                        learning_rate=0.01, epochs=100):
                super(GRURegressor, self).__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.batch_size = batch_size
                self.dropout = dropout
                self.layer_size = layer_size
                self.num_layers = num_layers
                self.learning_rate = learning_rate
                self.epochs = epochs
                
                self.gru = nn.GRU(input_dim, layer_size, num_layers, 
                                batch_first=True, dropout=dropout,
                                bidirectional=False)
                self.fc = nn.Linear(layer_size, output_dim)
                self.bn = nn.BatchNorm1d(layer_size)

            def forward(self, x):
                x = x.unsqueeze(1)  # Add sequence length dimension
                _, hidden = self.gru(x)
                hidden = hidden[-1]
                # hidden = self.bn(hidden)  # Apply batch normalization
                x = self.fc(hidden)  # Use the last hidden state
                return x

            def fit(self, x, y):
                dataset = TensorDataset(x, y)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
                criterion = nn.MSELoss()
                for batch_inputs, batch_targets in dataloader:
                    optimizer.zero_grad()
                    predictions = self(batch_inputs)
                    loss = criterion(predictions, batch_targets)
                    loss.backward()
                    optimizer.step()

            def predict(self, x):
                with torch.no_grad():
                    x = x.unsqueeze(1)  # Add sequence length dimension
                    _, hidden = self.gru(x)
                    x = self.fc(hidden[-1])  # Use the last hidden state
                    return x.detach().numpy()

        

        torch.manual_seed(25)
        np.random.seed(25)
        
        #global best_dev_loss_ever
        #global best_params_ever
        best_dev_loss_ever = float('inf')
        best_params_ever = {}
                    
            # Objective function for Optuna
        def optuna_objective(trial):
            
            # Hyperparameters to tune
            if input_dim < 100:                 
                layer_size = trial.suggest_categorical('layer_size', [2, 5, 10, 100])
                learning_rate = trial.suggest_categorical('lr', [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
                dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.5])
                num_layers = trial.suggest_categorical('num_layers', [2, 3, 4])
                epochs = trial.suggest_int('epochs', 50, 300, step=50)
            else:
                layer_size = trial.suggest_int('layer_size', 500, 2000, step=500)
                learning_rate = trial.suggest_categorical('lr', [5e-4, 1e-4, 5e-5, 1e-5])
                dropout = trial.suggest_categorical("dropout", 0.3, 0.4, 0.5)
                num_layers = trial.suggest_categorical('num_layers', [2, 3, 4])
                epochs = trial.suggest_int('epochs', 100, 500, step=100)
            #nhead = trial.suggest_int("nhead", 1, 4)

            model = GRURegressor(input_dim, output_dim, 
                                            layer_size=layer_size, num_layers=num_layers, 
                                            dropout=dropout,
                                            learning_rate=learning_rate,
                                            batch_size=batch_size, epochs=epochs)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            best_dev_loss = float('inf')

            # Training Loop
            for epoch in range(epochs): 
                model.train()
                epoch_loss = 0.0
                for batch_inputs, batch_targets in trainloader:
                    optimizer.zero_grad()
                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                # Evaluate on dev set
                model.eval()
                dev_loss = 0.0
                with torch.no_grad():
                    for batch_inputs, batch_targets in devloader:
                        outputs = model(batch_inputs)
                        outputs = outputs.view(batch_targets.shape)
                        loss = criterion(outputs, batch_targets)
                        dev_loss += loss.item()
                        
                    dev_loss = dev_loss / len(devloader)
                    best_dev_loss = min(best_dev_loss, dev_loss)
                    
                    global best_dev_loss_ever
                    if best_dev_loss < best_dev_loss_ever:
                        best_dev_loss_ever = best_dev_loss
                        #print("Best dev loss:", best_dev_loss)
                        torch.save(model.state_dict(), '/data/smangalik/best_gru.pth')
                        global best_params_ever # write down the best hyperparameters
                        best_params_ever = {
                            'layer_size': layer_size,
                            'learning_rate': learning_rate,
                            'dropout': dropout,
                            'num_layers': num_layers,
                            'epochs': epochs
                        }
                
                    trial.report(dev_loss, step=epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
            return best_dev_loss  # Optuna minimizes this loss

        sampler = optuna.samplers.TPESampler(seed=25)
        study = optuna.create_study(
            study_name="GRU_Optuna",
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(max_resource=epochs)
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(optuna_objective, n_trials=20, show_progress_bar=True)

        # print("Best hyperparameters:", study.best_params, "Best loss:", study.best_value)   
        # layer_size = study.best_params['layer_size']
        # learning_rate = study.best_params['lr']
        # dropout = study.best_params['dropout']
        # num_layers = study.best_params['num_layers']
        # epochs = study.best_params['epochs']
        
        print("Best hyperparameters:", best_params_ever, "Best loss:", best_dev_loss_ever)
        layer_size = best_params_ever['layer_size']
        learning_rate = best_params_ever['learning_rate']
        dropout = best_params_ever['dropout']
        num_layers = best_params_ever['num_layers']
        epochs = best_params_ever['epochs']

        # Run the model with the best hyperparameters
        model = GRURegressor(input_dim, output_dim, 
                            layer_size=layer_size, num_layers=num_layers, 
                            dropout=dropout, learning_rate=learning_rate, 
                            batch_size=batch_size, epochs=epochs)
        
        model.load_state_dict(torch.load('/data/smangalik/best_gru.pth'))
        #model.fit(X_tr_tensor, y_tr_tensor)
        
        y_pred = model.predict(X_test_tensor)          
        
    elif model_name == 'GRU_GPU':
         # Ensure tensors are moved to the GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        # Default Hyperparameters
        epochs = 200
        batch_size = 64

        # Tunable Hyperparameters
        learning_rate = 0.01
        dropout = 0.1
        layer_size = 100
        num_layers = 2

        # Create a train and dev split
        X_tr_tensor, X_dev_tensor, y_tr_tensor, y_dev_tensor = train_test_split(
            X_train_tensor, y_train_tensor, test_size=0.25, random_state=25)

        trainloader = DataLoader(TensorDataset(X_tr_tensor, y_tr_tensor), batch_size=batch_size, shuffle=True)
        devloader = DataLoader(TensorDataset(X_dev_tensor, y_dev_tensor), batch_size=batch_size, shuffle=False)

        class GRURegressor(nn.Module):
            def __init__(self, input_dim, output_dim, 
                        layer_size=100, num_layers=1, 
                        dropout=0.1, batch_size=64, 
                        learning_rate=0.01, epochs=100, device=None):
                super(GRURegressor, self).__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.batch_size = batch_size
                self.dropout = dropout
                self.layer_size = layer_size
                self.num_layers = num_layers
                self.learning_rate = learning_rate
                self.epochs = epochs
                
                # Set device
                self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

                # Move model to device
                self.gru = nn.GRU(input_dim, layer_size, num_layers, 
                                batch_first=True, dropout=dropout,
                                bidirectional=False).to(self.device)
                self.fc = nn.Linear(layer_size, output_dim).to(self.device)
                self.bn = nn.BatchNorm1d(layer_size).to(self.device)

            def forward(self, x):
                x = x.to(self.device).unsqueeze(1)  # Move to device and add sequence length dimension
                _, hidden = self.gru(x)
                hidden = hidden[-1]
                x = self.fc(hidden)  # Use the last hidden state
                return x

            def fit(self, x, y):
                x, y = x.to(self.device), y.to(self.device)  # Move data to device
                dataset = TensorDataset(x, y)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
                criterion = nn.MSELoss()

                self.train()
                for batch_inputs, batch_targets in dataloader:
                    batch_inputs, batch_targets = batch_inputs.to(self.device), batch_targets.to(self.device)  # Move batch to device
                    optimizer.zero_grad()
                    predictions = self(batch_inputs)
                    loss = criterion(predictions, batch_targets)
                    loss.backward()
                    optimizer.step()

            def predict(self, x):
                x = x.to(self.device).unsqueeze(1)  # Move to device and add sequence length dimension
                with torch.no_grad():
                    _, hidden = self.gru(x)
                    x = self.fc(hidden[-1])  # Use the last hidden state
                return x.cpu().detach().numpy()  # Move back to CPU for numpy conversion

        # Move the model to GPU
        model = GRURegressor(input_dim, output_dim, layer_size=layer_size, num_layers=num_layers, dropout=dropout, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs).to(device)
        
        # Training process and Optuna integration remain unchanged but ensure tensors and model are moved to `device`
        torch.manual_seed(25)
        np.random.seed(25)
        
        #global best_dev_loss_ever
        #global best_params_ever
        best_dev_loss_ever = float('inf')
        best_params_ever = {}
                    
            # Objective function for Optuna
        def optuna_objective(trial):
            
            # Hyperparameters to tune
            if input_dim < 100:                 
                layer_size = trial.suggest_categorical('layer_size', [5, 10, 100])
                learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 5e-2, 5e-3, 1e-3, 5e-4, 1e-4])
                dropout = trial.suggest_categorical("dropout", [0.3, 0.4, 0.5])
                num_layers = trial.suggest_categorical('num_layers', [2, 3])
                epochs = trial.suggest_int('epochs', 50, 200, step=50)
            else:
                layer_size = trial.suggest_categorical('layer_size', [500])
                learning_rate = trial.suggest_categorical('lr', [1e-4, 5e-4, 1e-4, 5e-5, 1e-5])
                dropout = trial.suggest_categorical("dropout", [0.5])
                num_layers = trial.suggest_categorical('num_layers', [2])
                epochs = trial.suggest_int('epochs', 25, 100, step=25)
            #nhead = trial.suggest_int("nhead", 1, 4)

            model = GRURegressor(input_dim, output_dim, 
                                            layer_size=layer_size, num_layers=num_layers, 
                                            dropout=dropout,
                                            learning_rate=learning_rate,
                                            batch_size=batch_size, epochs=epochs)
            model.to(device)  # Move model to device
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            criterion = nn.MSELoss()
            
            best_dev_loss = float('inf')

            # Training Loop
            for epoch in range(epochs): 
                model.train()
                epoch_loss = 0.0
                for batch_inputs, batch_targets in trainloader:
                    optimizer.zero_grad()
                    outputs = model(batch_inputs).to(device)  # Move batch to device
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                # Evaluate on dev set
                model.eval()
                dev_loss = 0.0
                with torch.no_grad():
                    for batch_inputs, batch_targets in devloader:
                        outputs = model(batch_inputs)
                        outputs = outputs.view(batch_targets.shape)
                        loss = criterion(outputs, batch_targets)
                        dev_loss += loss.item()
                        
                    dev_loss = dev_loss / len(devloader)
                    best_dev_loss = min(best_dev_loss, dev_loss)
                    
                    global best_dev_loss_ever
                    if best_dev_loss < best_dev_loss_ever:
                        best_dev_loss_ever = best_dev_loss
                        #print("Best dev loss:", best_dev_loss)
                        torch.save(model.state_dict(), '/data/smangalik/best_gru.pth')
                        global best_params_ever # write down the best hyperparameters
                        best_params_ever = {
                            'layer_size': layer_size,
                            'learning_rate': learning_rate,
                            'dropout': dropout,
                            'num_layers': num_layers,
                            'epochs': epochs
                        }
                
                    trial.report(dev_loss, step=epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
            return best_dev_loss  # Optuna minimizes this loss

        sampler = optuna.samplers.TPESampler(seed=25)
        study = optuna.create_study(
            study_name="GRU_Optuna",
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(max_resource=epochs)
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(optuna_objective, n_trials=20, show_progress_bar=True)

        # print("Best hyperparameters:", study.best_params, "Best loss:", study.best_value)   
        # layer_size = study.best_params['layer_size']
        # learning_rate = study.best_params['lr']
        # dropout = study.best_params['dropout']
        # num_layers = study.best_params['num_layers']
        # epochs = study.best_params['epochs']
        
        print("Best hyperparameters:", best_params_ever, "Best loss:", best_dev_loss_ever)
        layer_size = best_params_ever['layer_size']
        learning_rate = best_params_ever['learning_rate']
        dropout = best_params_ever['dropout']
        num_layers = best_params_ever['num_layers']
        epochs = best_params_ever['epochs']

        # Run the model with the best hyperparameters
        model = GRURegressor(input_dim, output_dim, 
                            layer_size=layer_size, num_layers=num_layers, 
                            dropout=dropout, learning_rate=learning_rate, 
                            batch_size=batch_size, epochs=epochs)
        
        model.load_state_dict(torch.load('/data/smangalik/best_gru.pth'))
        #model.fit(X_tr_tensor, y_tr_tensor)
        
        y_pred = model.predict(X_test_tensor)          
              
    elif model_name == 'Transformer':
        
        # Default Hyperparameters
        epochs = 200 
        batch_size = 16
        
        # Tunable Hyperparameters
        learning_rate = 0.01
        dropout = 0.1
        layer_size = 100
        num_layers = 2 
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        # Create a train and dev split
        X_tr_tensor, X_dev_tensor, y_tr_tensor, y_dev_tensor = train_test_split(
                X_train_tensor, y_train_tensor, test_size=0.25, random_state=25)
        
        trainloader = DataLoader(TensorDataset(X_tr_tensor, y_tr_tensor), batch_size=batch_size, shuffle=True)
        devloader = DataLoader(TensorDataset(X_dev_tensor, y_dev_tensor), batch_size=batch_size, shuffle=False)

        
        class TransformerRegressor(nn.Module):
            def __init__(self, input_dim, output_dim,
                            layer_size=100, num_layers=1,
                            dropout=0.1, batch_size=64, 
                            learning_rate=0.01, epochs=100):
                super(TransformerRegressor, self).__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.batch_size = batch_size
                self.num_heads = max([i for i in range(1, input_dim) if input_dim % i == 0 and i <= input_dim])
                self.dropout = dropout
                self.layer_size = layer_size
                self.num_layers = num_layers
                self.learning_rate = learning_rate
                self.epochs = epochs
                self.encoder_layer = nn.TransformerEncoderLayer(
                    d_model=input_dim, nhead=self.num_heads, dim_feedforward=layer_size, dropout=dropout)
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
                self.fc = nn.Linear(input_dim, output_dim)
            
            
            def forward(self, x):
                x = x.unsqueeze(0)  # Add batch dimension
                # pad the input to match the expected input size of the transformer
                x = self.transformer_encoder(x)
                x = x.mean(dim=0)  # Average over the sequence length
                x = self.fc(x)
                return x
            
            def fit(self, x, y):
                dataset = TensorDataset(x, y)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
                criterion = nn.MSELoss()
                for batch_inputs, batch_targets in dataloader:
                    optimizer.zero_grad()
                    predictions = self(batch_inputs)
                    loss = criterion(predictions, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
            def predict(self, x):
                with torch.no_grad():
                    x = x.unsqueeze(0)  # Add batch dimension
                    x = self.transformer_encoder(x)
                    x = x.mean(dim=0)  # Average over the sequence length
                    x = self.fc(x)
                    return x.detach().numpy()
        
        torch.manual_seed(25)
        np.random.seed(25)
        
        best_dev_loss_ever = float('inf')
        best_params_ever = {}
                    
        # Objective function for Optuna
        def optuna_objective(trial):
            
            # Hyperparameters to tune                
            
            if input_dim < 100:                 
                layer_size = trial.suggest_categorical('layer_size', [5, 10])
                learning_rate = trial.suggest_categorical('learning_rate', [1e-3, 5e-3, 1e-3])
                dropout = trial.suggest_categorical("dropout", [0.5])
                num_layers = trial.suggest_categorical('num_layers', [2])
                epochs = trial.suggest_int('epochs', 50, 200, step=25)
            else:
                layer_size = trial.suggest_int('layer_size', 500, 2000, step=500)
                learning_rate = trial.suggest_categorical('lr', [5e-4, 1e-4, 5e-5, 1e-5])
                dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
                num_layers = trial.suggest_categorical('num_layers', [2, 4])
                epochs = trial.suggest_int('epochs', 100, 500, step=100)
            #nhead = trial.suggest_int("nhead", 1, 4)

            model = TransformerRegressor(input_dim=input_dim, output_dim=output_dim, 
                                            layer_size=layer_size, num_layers=num_layers, 
                                            dropout=dropout,
                                            learning_rate=learning_rate,
                                            batch_size=batch_size, epochs=epochs)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            
            # Training Loop
            best_dev_loss = float('inf')
            for epoch in range(epochs): 
                model.train()
                epoch_loss = 0.0
                for batch_inputs, batch_targets in trainloader:
                    optimizer.zero_grad()
                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                dev_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for batch_inputs, batch_targets in devloader:
                        outputs = model(batch_inputs)
                        outputs = outputs.view(batch_targets.shape)
                        loss = criterion(outputs, batch_targets)
                        dev_loss += loss.item()
                        
                    dev_loss = dev_loss / len(devloader)
                    best_dev_loss = min(best_dev_loss, dev_loss)
                    
                    global best_dev_loss_ever
                    if best_dev_loss < best_dev_loss_ever:
                        best_dev_loss_ever = best_dev_loss
                        #print("Best dev loss:", best_dev_loss)
                        torch.save(model.state_dict(), '/data/smangalik/best_transformer.pth')
                        global best_params_ever
                        best_params_ever = {
                            'layer_size': layer_size,
                            'learning_rate': learning_rate,
                            'dropout': dropout,
                            'num_layers': num_layers,
                            'epochs': epochs
                        }
                    
                trial.report(dev_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
            return best_dev_loss

        sampler = optuna.samplers.TPESampler(seed=25)
        study = optuna.create_study(
            study_name="Transformer_Optuna",
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(max_resource=epochs)
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(optuna_objective, n_trials=20, show_progress_bar=True)

        # print("Best hyperparameters:", study.best_params, "Best loss:", study.best_value)   
        # layer_size = study.best_params['layer_size']
        # learning_rate = study.best_params['learning_rate']
        # dropout = study.best_params['dropout']
        # num_layers = study.best_params['num_layers']
        # epochs = study.best_params['epochs']
        
        print("Best hyperparameters:", best_params_ever, "Best loss:", best_dev_loss_ever)
        layer_size = best_params_ever['layer_size']
        learning_rate = best_params_ever['learning_rate']
        dropout = best_params_ever['dropout']
        num_layers = best_params_ever['num_layers']
        epochs = best_params_ever['epochs']
        
        # Run the model with the best hyperparameters
        model = TransformerRegressor(input_dim, output_dim, 
                                        layer_size=layer_size, num_layers=num_layers, 
                                        dropout=dropout,
                                        learning_rate=learning_rate,
                                        batch_size=batch_size, epochs=epochs)
        
        model.load_state_dict(torch.load('/data/smangalik/best_transformer.pth'))
        #model.fit(X_tr_tensor, y_tr_tensor)
        
        y_pred = model.predict(X_test_tensor)                                       
    
    elif model_name == 'Transformer_GPU':
        # Ensure tensors and model are moved to the GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default Hyperparameters
        epochs = 200
        batch_size = 64
        nhead = 1

        # Tunable Hyperparameters
        learning_rate = 0.01
        dropout = 0.1
        layer_size = 100
        num_layers = 2

        # Convert data to PyTorch tensors and move to GPU
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        # Create a train and dev split
        X_tr_tensor, X_dev_tensor, y_tr_tensor, y_dev_tensor = train_test_split(
            X_train_tensor, y_train_tensor, test_size=0.25, random_state=25)

        trainloader = DataLoader(TensorDataset(X_tr_tensor, y_tr_tensor), batch_size=batch_size, shuffle=True)
        devloader = DataLoader(TensorDataset(X_dev_tensor, y_dev_tensor), batch_size=batch_size, shuffle=False)

        class TransformerRegressor(nn.Module):
            def __init__(self, input_dim, output_dim, layer_size=100, num_layers=1,
                        num_heads=1, dropout=0.1, batch_size=64, learning_rate=0.01, epochs=100):
                super(TransformerRegressor, self).__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.batch_size = batch_size
                self.num_heads = num_heads
                self.dropout = dropout
                self.layer_size = layer_size
                self.num_layers = num_layers
                self.learning_rate = learning_rate
                self.epochs = epochs
                
                self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

                self.encoder_layer = nn.TransformerEncoderLayer(
                    d_model=input_dim, nhead=num_heads, dim_feedforward=layer_size, dropout=dropout
                ).to(self.device)
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers).to(self.device)
                self.fc = nn.Linear(input_dim, output_dim).to(self.device)

            def forward(self, x):
                x = x.to(self.device).unsqueeze(0)  # Add batch dimension
                x = self.transformer_encoder(x)
                x = x.mean(dim=0)  # Average over the sequence length
                x = self.fc(x)
                return x

            def fit(self, x, y):
                x, y = x.to(self.device), y.to(self.device)  # Move data to device
                dataset = TensorDataset(x, y)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
                criterion = nn.MSELoss()
                self.train()
                for batch_inputs, batch_targets in dataloader:
                    batch_inputs, batch_targets = batch_inputs.to(self.device), batch_targets.to(self.device)
                    optimizer.zero_grad()
                    predictions = self(batch_inputs)
                    loss = criterion(predictions, batch_targets)
                    loss.backward()
                    optimizer.step()

            def predict(self, x):
                with torch.no_grad():
                    x = x.unsqueeze(0)  # Add batch dimension
                    x = self.transformer_encoder(x.to(self.device))
                    x = x.mean(dim=0)  # Average over the sequence length
                    x = self.fc(x)
                    return x.cpu().detach().numpy()

        # Move the model to GPU
        model = TransformerRegressor(input_dim, output_dim, layer_size=layer_size, num_layers=num_layers,
                                    num_heads=nhead, dropout=dropout, learning_rate=learning_rate,
                                    batch_size=batch_size, epochs=epochs).to(device)

        # Ensure Optuna training loops use GPU tensors
        torch.manual_seed(25)
        np.random.seed(25)
        
        best_dev_loss_ever = float('inf')
        best_params_ever = {}
                    
        # Objective function for Optuna
        def optuna_objective(trial):
            
            # Hyperparameters to tune                
            if input_dim < 100:                 
                layer_size = trial.suggest_categorical('layer_size', [5, 10, 100])
                learning_rate = trial.suggest_categorical('learning_rate', [1e-2, 5e-2, 5e-3, 1e-3, 5e-4, 1e-4])
                dropout = trial.suggest_categorical("dropout", [0.5])
                num_layers = trial.suggest_categorical('num_layers', [2, 3, 4])
                epochs = trial.suggest_int('epochs', 50, 200, step=50)
            else:
                layer_size = trial.suggest_categorical('layer_size', [500])
                learning_rate = trial.suggest_categorical('lr', [0.0001])
                dropout = trial.suggest_categorical("dropout", [0.1])
                num_layers = trial.suggest_categorical('num_layers', [2])
                epochs = trial.suggest_categorical('epochs', [25,50,75,100])
            #nhead = trial.suggest_int("nhead", 1, 4)

            model = TransformerRegressor(input_dim=input_dim, output_dim=output_dim, 
                                            layer_size=layer_size, num_layers=num_layers, 
                                            num_heads=nhead, dropout=dropout,
                                            learning_rate=learning_rate,
                                            batch_size=batch_size, epochs=epochs)
            model.to(device)  # Move model to device
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            
            # Training Loop
            best_dev_loss = float('inf')
            for epoch in range(epochs): 
                model.train()
                epoch_loss = 0.0
                for batch_inputs, batch_targets in trainloader:
                    optimizer.zero_grad()
                    outputs = model(batch_inputs).to(device)  # Move batch to device
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                dev_loss = 0.0
                model.eval()
                with torch.no_grad():
                    for batch_inputs, batch_targets in devloader:
                        outputs = model(batch_inputs)
                        outputs = outputs.view(batch_targets.shape)
                        loss = criterion(outputs, batch_targets)
                        dev_loss += loss.item()
                        
                    dev_loss = dev_loss / len(devloader)
                    best_dev_loss = min(best_dev_loss, dev_loss)
                    
                    global best_dev_loss_ever
                    if best_dev_loss < best_dev_loss_ever:
                        best_dev_loss_ever = best_dev_loss
                        #print("Best dev loss:", best_dev_loss)
                        torch.save(model.state_dict(), '/data/smangalik/best_transformer.pth')
                        global best_params_ever
                        best_params_ever = {
                            'layer_size': layer_size,
                            'learning_rate': learning_rate,
                            'dropout': dropout,
                            'num_layers': num_layers,
                            'epochs': epochs
                        }
                    
                trial.report(dev_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
            return best_dev_loss

        sampler = optuna.samplers.TPESampler(seed=25)
        study = optuna.create_study(
            study_name="Transformer_Optuna",
            direction="minimize",
            sampler=sampler,
            pruner=optuna.pruners.HyperbandPruner(max_resource=epochs)
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(optuna_objective, n_trials=20, show_progress_bar=True)

        # print("Best hyperparameters:", study.best_params, "Best loss:", study.best_value)   
        # layer_size = study.best_params['layer_size']
        # learning_rate = study.best_params['learning_rate']
        # dropout = study.best_params['dropout']
        # num_layers = study.best_params['num_layers']
        # epochs = study.best_params['epochs']
        
        print("Best hyperparameters:", best_params_ever, "Best loss:", best_dev_loss_ever)
        layer_size = best_params_ever['layer_size']
        learning_rate = best_params_ever['learning_rate']
        dropout = best_params_ever['dropout']
        num_layers = best_params_ever['num_layers']
        epochs = best_params_ever['epochs']
        
        # Run the model with the best hyperparameters
        model = TransformerRegressor(input_dim, output_dim, 
                                        layer_size=layer_size, num_layers=num_layers, 
                                        num_heads=nhead, dropout=dropout,
                                        learning_rate=learning_rate,
                                        batch_size=batch_size, epochs=epochs)
        
        model.load_state_dict(torch.load('/data/smangalik/best_transformer.pth'))
        #model.fit(X_tr_tensor, y_tr_tensor)
        
        y_pred = model.predict(X_test_tensor)       
    
    else:
        print("Unknown model:", model_name)
        return None, None
    
    # Column-wise MSE
    for i, y_col in enumerate(outcomes):
        y_pred_col = y_pred[:, i]
        y_test_col = y_test[:, i]
        mse = np.mean((y_pred_col - y_test_col) ** 2)
        if np.std(y_pred_col) == 0: # if preds are all the same, set corr to 0
            corr = 0.0
        else:
            corr = np.corrcoef(y_pred_col, y_test_col)[0, 1]
        print(f"-> {y_col.replace(' ', '_')} MSE: {round(mse, 3):.3f} \tCorr: {round(corr, 3):.3f}")
        
    residuals = abs(y_test - y_pred)
    if model_name in ['RidgeRegression','RandomForest','XGBoost','MLPRegressor']:
        return grid_search.best_estimator_, residuals
    elif model_name in ['KNeighborsRegressor', 'FFN', 'GRU', 'GRU_GPU', 'Transformer', 'Transformer_GPU']:
        return model, residuals
    else:
        return None, residuals
    
    
def p_value_evaluate(res_better, res_control): 
   
    assert res_better.shape == res_control.shape, "Residuals must be of the same shape"
    n = len(res_better)
    p_vals = []
    for i, y_col in enumerate(range(res_better.shape[1])):
        y_fcra_diff = res_better[:, i] - res_control[:, i]
        y_fcra_diff_mean= np.mean(y_fcra_diff)
        y_fcra_sd = np.std(y_fcra_diff)
        y_fcra_diff_t = abs( y_fcra_diff_mean / (y_fcra_sd / np.sqrt(n)) )
        y_fcra_diff_p = stats.t.sf( y_fcra_diff_t, df = n-1 )
        p_vals.append(y_fcra_diff_p)
    return p_vals

    
##################################
# Put your experiments below!
#################################
    
    
# Guess that no change will happen
evaluate(X, y, model_name='NoChange', run_name="No Change Baseline")

# Always guess the mean value for each column
_, residuals_mean_baseline = evaluate(X, y, model_name='MeanBaseline', run_name="Mean Baseline")

# Only use the P
_, residuals_ridge_p = evaluate(X, y, model_name="RidgeRegression", run_name="P and Ridge")
_, residuals_ridge_rc = evaluate(X_line_params, y, model_name="RidgeRegression", run_name="RC and Ridge")
_, residuals_ridge_p_rc = evaluate(X_with_line_params, y, model_name="RidgeRegression", run_name="P + RC and Ridge")
evaluate(X_with_cov, y, model_name="RidgeRegression", run_name="P + Cov and Ridge")
evaluate(X_with_line_params_and_cov, y, model_name="RidgeRegression", run_name="P + RC + Cov and Ridge")
_, residuals_ridge_p_rc_cov = evaluate(X_with_line_params_and_cov_and_cov_line_params, y, model_name="RidgeRegression", run_name="P + RC + Cov + Cov RC and Ridge")

_, residuals_ridge_e = evaluate(X_embedding, y, model_name="RidgeRegression", run_name="E and Ridge")
_, residuals_ridge_p_e = evaluate(X_with_embedding, y, model_name="RidgeRegression", run_name="P + E and Ridge")
_, residuals_ridge_rc_e = evaluate(X_line_params_with_embedding, y, model_name="RidgeRegression", run_name="RC + E and Ridge")
_, residuals_ridge_p_rc_e = evaluate(X_with_line_params_and_embedding, y, model_name="RidgeRegression", run_name="P + RC + E and Ridge")
_, residuals_ridge_p_rc_cov = evaluate(X_with_line_params_and_cov_and_cov_line_params, y, model_name="RidgeRegression", run_name="P + RC + Cov + Cov RC and Ridge")
_, residuals_ridge_p_rc_every = model, residuals = evaluate(X_everything, y, model_name='RidgeRegression', run_name="Everything and Ridge")

# Test the P + LP model across SES3 and urbanicity
df['lnurban3'] = pd.qcut(df['lnurban'], 3, labels=["Rural", "Suburban", "Urban"])
pred_df = pd.DataFrame(model.predict(X_everything), columns=['pred b', 'pred m'])
pred_df = pd.concat([pred_df, y], axis=1)
pred_df = pred_df.merge(df, on='county')
print()
for ses in pred_df['ses3'].unique():
    dat = pred_df[pred_df['ses3'] == ses]
    #print(f"SES3: {ses} -> Count: {len(dat)}")
    # Calculate the MSE and correlation for each SES3 group
    mse_b = np.mean((dat['pred b'] - dat['Delta b']) ** 2)
    mse_m = np.mean((dat['pred m'] - dat['Delta m']) ** 2)
    corr_b = np.corrcoef(dat['pred b'], dat['Delta b'])[0, 1]
    corr_m = np.corrcoef(dat['pred m'], dat['Delta m'])[0, 1]
    print(f"SES{ses} (n={len(dat)})): -> Delta b MSE: {round(mse_b, 3)} \tDelta b Corr: {round(corr_b, 3)}")
    print(f"SES{ses} (n={len(dat)})): -> Delta m MSE: {round(mse_m, 3)} \tDelta m Corr: {round(corr_m, 3)}")
    # Mean and std of the predictions and true values
    print(f"SES{ses} (n={len(dat)})): -> Delta b Mean: {round(np.mean(dat['pred b']), 3)} \tDelta b Std: {round(np.std(dat['pred b']), 3)}")
    print(f"SES{ses} (n={len(dat)})): -> Delta m Mean: {round(np.mean(dat['pred m']), 3)} \tDelta m Std: {round(np.std(dat['pred m']), 3)}")
print()
for urbanicity in pred_df['lnurban3'].unique():
    dat = pred_df[pred_df['lnurban3'] == urbanicity]
    mse_b = np.mean((dat['pred b'] - dat['Delta b']) ** 2)
    mse_m = np.mean((dat['pred m'] - dat['Delta m']) ** 2)
    corr_b = np.corrcoef(dat['pred b'], dat['Delta b'])[0, 1]
    corr_m = np.corrcoef(dat['pred m'], dat['Delta m'])[0, 1]
    print(f"{urbanicity} (n={len(dat)})): -> Delta b MSE: {round(mse_b, 3)} \tDelta b Corr: {round(corr_b, 3)}")
    print(f"{urbanicity} (n={len(dat)})): -> Delta m MSE: {round(mse_m, 3)} \tDelta m Corr: {round(corr_m, 3)}")
    # Mean and std of the predictions and true values
    print(f"{urbanicity} (n={len(dat)})): -> Delta b Mean: {round(np.mean(dat['pred b']), 3)} \tDelta b Std: {round(np.std(dat['pred b']), 3)}")
    print(f"{urbanicity} (n={len(dat)})): -> Delta m Mean: {round(np.mean(dat['pred m']), 3)} \tDelta m Std: {round(np.std(dat['pred m']), 3)}")

# K-Nearest Neighbors (KNN)
_, residuals_knn_p_rc = evaluate(X_with_line_params, y, model_name="KNeighborsRegressor", run_name="P + RC and KNeighborsRegressor")
_, residuals_knn_p_rc_cov = evaluate(X_with_line_params_and_cov_and_cov_line_params, y, model_name="KNeighborsRegressor", run_name="P + RC + Cov + Cov RC and KNeighborsRegressor")
_, residuals_knn_p_rc_e = evaluate(X_with_line_params_and_embedding, y, model_name="KNeighborsRegressor", run_name="E + P + RC and KNeighborsRegressor")
_, residuals_knn_p_rc_every = evaluate(X_everything, y, model_name="KNeighborsRegressor", run_name="Everything and KNeighborsRegressor")

# Feed Forward Neural Network (FFN)
_, residuals_fnn_p_rc = evaluate(X_with_line_params, y, model_name="FFN", run_name="P + RC and FFN")
evaluate(X_with_line_params_and_embedding, y, model_name="FFN", run_name="E + P + RC and FFN")
evaluate(X_with_line_params_and_cov_and_cov_line_params, y, model_name="FFN", run_name="P + RC + Cov + Cov RC and FFN")
evaluate(X_everything, y, model_name="FFN", run_name="Everything and FFN")

# GRU Model
_, residuals_gru_p_rc = evaluate(X_with_line_params, y, model_name="GRU", run_name="P + RC and GRU")
evaluate(X_with_line_params_and_cov_and_cov_line_params, y, model_name="GRU_GPU", run_name="P + RC + Cov + Cov RC and GRU (on GPU)")
evaluate(X_with_line_params_and_embedding, y, model_name="GRU_GPU", run_name="E + P + RC and GRU (on GPU)")
_, residuals_gru_p_rc_every = evaluate(X_everything, y, model_name="GRU_GPU", run_name="Everything and GRU (on GPU)")

# Transformer Mode
_, residuals_trans_p_rc = evaluate(X_with_line_params, y, model_name="Transformer", run_name="P + RC and Transformer")
evaluate(X_with_line_params_and_cov_and_cov_line_params, y, model_name="Transformer_GPU", run_name="P + RC + Cov + Cov RC and Transformer (on GPU)")
evaluate(X_with_line_params_and_embedding, y, model_name="Transformer_GPU", run_name="E + P + RC and Transformer (on GPU)")
_, residuals_trans_p_rc_every = evaluate(X_everything, y, model_name="Transformer_GPU", run_name="Everything and Transformer (on GPU)")

# RandomForest
_, residuals_rf_p_rc = evaluate(X_with_line_params, y, model_name="RandomForest", run_name="P + RC and RandomForest")
evaluate(X_with_line_params_and_embedding, y, model_name="RandomForest", run_name="E + P + RC and RandomForest")

# # Use the P and the RC and XGBoost
_, residuals_xgb_p_rc = evaluate(X_with_line_params, y, model_name="XGBoost", run_name="P + RC and XGBoost")
evaluate(X_with_line_params_and_embedding, y, model_name="XGBoost", run_name="E + P + RC and XGBoost")

# Use the P and the RC and Extra Trees
_, residuals_et_p_rc =  evaluate(X_with_line_params, y, model_name="ExtraTrees", run_name="P + RC and Extra Trees")
_, residuals_et_p_rc_cov = evaluate(X_with_line_params_and_cov_and_cov_line_params, y, model_name="ExtraTrees", run_name="P + RC + Cov + Cov RC and Extra Trees")
_, residuals_et_p_rc_e = evaluate(X_with_line_params_and_embedding, y, model_name="ExtraTrees", run_name="E + P + RC and Extra Trees")
_, residuals_et_p_rc_every = evaluate(X_everything, y, model_name="ExtraTrees", run_name="Everything and Extra Trees")

# # Use the P and the RC and sklearn MLP
# evaluate(X_with_line_params, y, model_name="MLPRegressor", run_name="RC and sklearn MLP")
# evaluate(X_with_line_params_and_embedding, y, model_name="MLPRegressor", run_name="E + P + LP and sklearn MLP")

# Calculate p-values for the residuals

print("\nTABLE 1")
print("Mean vs Ridge", p_value_evaluate(residuals_ridge_p_rc, residuals_mean_baseline))
print("Mean vs KNN", p_value_evaluate(residuals_knn_p_rc, residuals_mean_baseline))
print("Mean vs FFN", p_value_evaluate(residuals_fnn_p_rc, residuals_mean_baseline))
print("Mean vs RandomForest", p_value_evaluate(residuals_rf_p_rc, residuals_mean_baseline))
print("Mean vs ExtraTrees", p_value_evaluate(residuals_et_p_rc, residuals_mean_baseline))
print("Mean vs XGBoost", p_value_evaluate(residuals_xgb_p_rc, residuals_mean_baseline))

print("\nTABLE 2")
print("Mean vs Ridge", p_value_evaluate(residuals_ridge_p_rc, residuals_mean_baseline))
print("Ridge vs Ridge + Cov", p_value_evaluate(residuals_ridge_p_rc_cov, residuals_ridge_p_rc))
print("Ridge vs Ridge + E", p_value_evaluate(residuals_ridge_p_rc_e, residuals_ridge_p_rc))
print("Ridge vs Ridge + Everything", p_value_evaluate(residuals_ridge_p_rc_every, residuals_ridge_p_rc))
print("Mean vs KNN", p_value_evaluate(residuals_knn_p_rc, residuals_mean_baseline))
print("KNN vs KNN + Cov", p_value_evaluate(residuals_knn_p_rc_cov, residuals_knn_p_rc))
print("KNN vs KNN + E", p_value_evaluate(residuals_knn_p_rc_e, residuals_knn_p_rc))
print("KNN vs KNN + Everything", p_value_evaluate(residuals_knn_p_rc_every, residuals_knn_p_rc))
print("Mean vs ExtraTrees", p_value_evaluate(residuals_et_p_rc, residuals_mean_baseline))
print("ExtraTrees vs ExtraTrees + Cov", p_value_evaluate(residuals_et_p_rc_cov, residuals_et_p_rc))
print("ExtraTrees vs ExtraTrees + E", p_value_evaluate(residuals_et_p_rc_e, residuals_et_p_rc))
print("ExtraTrees vs ExtraTrees + Everything", p_value_evaluate(residuals_et_p_rc_every, residuals_et_p_rc))

print("\nTABLE 3")
print("Mean vs ExtraTrees", p_value_evaluate(residuals_et_p_rc, residuals_mean_baseline))
print("ExtraTrees vs ExtraTrees + Everything", p_value_evaluate(residuals_et_p_rc_every, residuals_et_p_rc))
print("Mean vs Transformer", p_value_evaluate(residuals_trans_p_rc, residuals_mean_baseline))
print("Transformer vs Transformer + Everything", p_value_evaluate(residuals_trans_p_rc_every, residuals_trans_p_rc))
print("Mean vs GRU", p_value_evaluate(residuals_gru_p_rc, residuals_mean_baseline))
print("GRU vs GRU + Everything", p_value_evaluate(residuals_gru_p_rc_every, residuals_gru_p_rc))

print("\nTABLE 5")
print("Mean vs Ridge RC", p_value_evaluate(residuals_ridge_rc, residuals_mean_baseline))
print("Mean vs Ridge P", p_value_evaluate(residuals_ridge_p, residuals_mean_baseline))
print("Mean vs Ridge P+RC", p_value_evaluate(residuals_ridge_p_rc, residuals_mean_baseline))
print("Mean vs Ridge E", p_value_evaluate(residuals_ridge_e, residuals_mean_baseline))
print("Mean vs Ridge E+RC", p_value_evaluate(residuals_ridge_rc_e, residuals_mean_baseline))
print("Mean vs Ridge E+P", p_value_evaluate(residuals_ridge_p_e, residuals_mean_baseline))
print("Mean vs Ridge P+RC+E", p_value_evaluate(residuals_ridge_p_rc_e, residuals_mean_baseline))
print("Mean vs Ridge P+RC+Cov", p_value_evaluate(residuals_ridge_p_rc_cov, residuals_mean_baseline))
print("Mean vs Ridge Everything", p_value_evaluate(residuals_ridge_p_rc_every, residuals_mean_baseline))
