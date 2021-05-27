import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from matplotlib.lines import Line2D
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from scipy.ndimage.filters import gaussian_filter
import io
import base64
from flask import current_app as app

def strike_zone_heatmaps(umpire, strikes, balls, outs, b_side, p_side):

    # baysian optimization
    def bayes_tuning(X, y, params, init_round=5, opt_round=5, random_state=42):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        def lgb_function(learning_rate, num_leaves, max_depth, min_child_samples):

            params['learning_rate'] = max(min(learning_rate, 1), 0)
            params['num_leaves'] = int(round(num_leaves))
            params['max_depth'] = int(round(max_depth))
            params['min_child_samples'] = int(round(min_child_samples))

            lgbc = lgb.LGBMClassifier(**params)
            lgbc.fit(X_train, y_train)
            y_pred = lgbc.predict(X_test)
            accuracy = precision_score(y_pred, y_test, average='micro')

            return accuracy

        lgbBO = BayesianOptimization(lgb_function, {'learning_rate': (0.01, 1.0), 'num_leaves': (30, 200),
                                                    'max_depth': (1, 30), 'min_child_samples': (20, 80)}, random_state=42)
        lgbBO.maximize(init_points=init_round, n_iter=opt_round)
        model_auc = []
        for model in range(len(lgbBO.res)):
            model_auc.append(lgbBO.res[model]['target'])

        return lgbBO.res[pd.Series(model_auc).idxmax()]['target'], lgbBO.res[pd.Series(model_auc).idxmax()]['params']

    # lightgbm model
    def lgb_classification(X, y, params, optimal_params):

        for param in optimal_params[1].keys():
            if param == 'learning_rate':
                optimal_params[1][param] = round(optimal_params[1][param], 2)
            else:
                optimal_params[1][param] = int(round(optimal_params[1][param]))
        params.update(optimal_params[1])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        lgbc = lgb.LGBMClassifier(**params)
        lgbc.fit(X_train, y_train)
        y_pred = lgbc.predict(X_test)
        accuracy = precision_score(y_pred, y_test, average='micro')
        cf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

        return lgbc, accuracy, cf_matrix 

    # load the data
    raw_data = pd.read_csv("{}/data/pitch_data_one_month.csv".format(app.root_path), index_col=0)
    data = raw_data[(raw_data['PITCH_RESULT'] == 'StrikeCalled') |
                    (raw_data['PITCH_RESULT'] == 'BallCalled')].reset_index(drop=True)

    # encoding and converting
    for feature in ['PITCH_RESULT', 'BATTER_SIDE', 'PITCHER_SIDE', 'PITCH_TYPE']:
        le = preprocessing.LabelEncoder()
        le.fit(data[feature])
        encoded_label = le.transform(data[feature])
        data[feature] = pd.DataFrame(encoded_label)

    categorical_features = ['CATCHER_ID', 'PITCHER_ID', 'BATTER_ID', 'UMPIRE_ID', 'BALLS',
                            'STRIKES', 'OUTS', 'BATTER_SIDE', 'PITCHER_SIDE', 'PITCH_TYPE']
    for feature in categorical_features:
        data[feature] = data[feature].astype("category")

    # input data
    def side_code(side):
        if side == 'Right':
            code = 1
        elif side == 'Left':
            code = 0
        return code

    x_grid = np.linspace(-2.5, 2.5, 500)
    y_grid = np.linspace(0, 5, 500)
    xx, yy = np.meshgrid(x_grid, y_grid)
    coords = np.array((xx.ravel(), yy.ravel())).T
    others = np.repeat(np.array([[umpire, balls, strikes, outs, side_code(b_side), side_code(p_side)]]),
                       500*500, axis=0)
    input_x = np.concatenate((others, coords), axis=1)

    # output data
    features0 = ['PITCH_LOCATION_SIDE', 'PITCH_LOCATION_HEIGHT']
    features1 = ['UMPIRE_ID', 'BALLS', 'STRIKES', 'OUTS', 'BATTER_SIDE',
                 'PITCHER_SIDE', 'PITCH_LOCATION_SIDE', 'PITCH_LOCATION_HEIGHT']

    results = []
    for feature, input_data in zip([features0, features1], [coords, input_x]):
        X = data[feature].to_numpy()
        y = data[['PITCH_RESULT']].to_numpy().ravel()
        params = {'boosting_type': 'gbdt', 'objective': 'binary'}
        optimal_params = bayes_tuning(X, y, params)
        lgbc_results = lgb_classification(X, y, params, optimal_params)
        result = lgbc_results[0].predict_proba(input_data)
        results.append(result[:, 1].reshape(500, 500))

    # visualization
    lines = [[(0.835, 1.5), (0.835, 3.5)], [(0.835, 3.5), (-0.835, 3.5)],
             [(-0.835, 3.5), (-0.835, 1.5)], [(-0.835, 1.5), (0.835, 1.5)]]
    fig1, axes = plt.subplots(ncols=2, figsize=(8, 4))
    for z, ax, title in zip(results, axes, ['LightGBM Model 1', 'LightGBM Model 2']):
        lc = mc.LineCollection(lines, linewidths=2.5, colors='k')
        smoothing_z = gaussian_filter(z, sigma=1.2)
        im = ax.imshow(smoothing_z, extent=[-2.5, 2.5, 0, 5], origin='lower', cmap='jet')
        ax.add_collection(lc)
        ax.axis('scaled')
        ax.set_xlim([-2.5, 2.5])
        ax.set_ylim([0, 5])
        ax.set_title(title)
        ax.set_xlabel('Pitch Location Side')
        ax.set_ylabel('Pitch Location Height')
    cax = fig1.add_axes([0.95, 0.15, 0.03, 0.7])
    fig1.colorbar(im, cax=cax)
    fig1.text(0.165, 0, '1. Model 1 only considers Pitch Location Side and Pitch Location Height as features.', ha='left')
    fig1.text(0.165, -0.08, '2. Model 2 considers Pitch Location Side, Pitch Location Height, Umpire ID, Strikes,\n    Balls, Outs, Batter Side, and Pitcher Side as features.', ha='left')

    save_file = io.BytesIO()
    fig1.savefig(save_file, format='png', bbox_inches='tight')
    figdata_png1 = base64.b64encode(save_file.getvalue()).decode('utf8')

    # visualization 2
    fig2, ax = plt.subplots(figsize=(3.65, 3.65))
    lc = mc.LineCollection(lines, linewidths=1.2, colors='k')
    plt.contour(xx, yy, gaussian_filter(results[0], sigma=1.6), levels = [0.5], colors='b')
    plt.contour(xx, yy, gaussian_filter(results[1], sigma=1.6), levels = [0.5], colors='r')
    ax.add_collection(lc)
    ax.axis('scaled')
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([0, 5])
    ax.set_title('50% Called Strike Probability')
    ax.set_xlabel('Pitch Location Side')
    ax.set_ylabel('Pitch Location Height')
    line = [Line2D([0], [0], color=c, linewidth=1.5) for c in ['b', 'r']]
    label = ['Model 1', 'Model 2']
    plt.legend(line, label)

    save_file = io.BytesIO()
    fig2.savefig(save_file, format='png', bbox_inches='tight')
    figdata_png2 = base64.b64encode(save_file.getvalue()).decode('utf8')

    return figdata_png1, figdata_png2
