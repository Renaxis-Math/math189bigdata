
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting functions

def plot_freq_distribution(data_list, bins_count, color):
    plt.hist(data_list, bins=bins_count, density=True, alpha=0.5, color = color)

    plt.title('Distribution plot')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    plt.show()
    
def plot_qq(residuals):
    sm.qqplot(residuals, line = 'q') 
    plt.title('Q-Q Plot')
    plt.show()

def plot_residual_vs_predicted(residuals, predicted_y):
    # Create the plot
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=predicted_y, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')

    plt.title('Residuals vs. Predicted')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    
    plt.show()


# Functions

def get_correlations(corr_df, intercept_col_name):
    copy_corr_df = corr_df.copy()
    if intercept_col_name in copy_corr_df: copy_corr_df.drop(intercept_col_name, axis = 1, inplace=True)
    
    answers = []

    corr_matrix = corr_df.values
    cols_count = len(corr_matrix)    
    for i in range(cols_count):
        for j in range(i+1, cols_count):
            corr_val = abs(round(corr_matrix[i, j], 2))
            answers.append(corr_val)
    
    return answers

def get_fully_correlated_pairs_set(corr_df):
    # Assuming 'df' is your correlation dataframe
    mask = np.ones(corr_df.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    filtered_corr_df = corr_df[mask & (corr_df == 1)]
    
    answers = set()

    for index_name in filtered_corr_df.index:
        for column_name in filtered_corr_df.columns:
            if not pd.isnull(filtered_corr_df.loc[index_name, column_name]):
                if ((index_name, column_name) not in answers) \
                    and ((column_name, index_name) not in answers):
                        answers.add((index_name, column_name))
    
    return answers

def inverse_boxcox(boxcox_y, lambda_val):
    if lambda_val == 0: return np.exp(boxcox_y)
    else: return np.power(lambda_val * boxcox_y + 1, 1 / lambda_val)
    
def vif_statsmodels(X, intercept_colname):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    
    X_copy = X.copy()
    if intercept_colname not in X_copy: X_copy = add_constant(X_copy)
    
    vif = pd.DataFrame()
    vif["variables"] = X_copy.columns
    vif["VIF"] = [variance_inflation_factor(X_copy.values, i) for i in range(X_copy.shape[1])]
    return vif
    
def vif_custom(X, lambdas, intercept_col_name):
    from numpy.linalg import inv
    copy_X = X.copy()
    if intercept_col_name in copy_X: copy_X.drop(intercept_col_name, axis=1, inplace=True)

    n_col = copy_X.shape[1] - 1
    vif = np.zeros((len(lambdas), n_col))
    beta = np.zeros((len(lambdas), n_col))

    # Compute correlation matrices
    rxx = copy_X.iloc[:, :n_col].corr().values
    rxy = copy_X.corr().iloc[:n_col, n_col].values

    for i in range(len(lambdas)):
        tmp1 = inv(rxx + lambdas[i] * np.eye(n_col))
        vif[i, :] = np.diag(np.dot(np.dot(tmp1, rxx), tmp1))
        beta[i, :] = np.dot(tmp1, rxy)

    ridge_beta_df = pd.DataFrame(beta, columns=copy_X.columns[:-1], index=lambdas)
    
    vif_df = pd.DataFrame(vif, columns=copy_X.columns[:-1], index=lambdas)
    return vif_df, ridge_beta_df

def find_optimal_ridge_alpha(vif_df, step_size):
    INVALID_LEFT = -2
    INVALID_RIGHT = 2
    
    def check(vif_df, left, right):
        lower_bound_condition = (vif_df.min(axis=COL) >= left)
        upper_bound_condition = (vif_df.max(axis=COL) <= right)
        
        return vif_df[lower_bound_condition & upper_bound_condition].shape[0] > 0
    
    from functools import lru_cache
    @lru_cache(maxsize = None)
    def solve(left, right):
        assert right >= left
        if right - left > 1: return (INVALID_LEFT, INVALID_RIGHT)
        if check(vif_df, left, right): return (left, right)
    
        left_answer = solve(left - step_size, right)
        left_range = left_answer[1] - left_answer[0]
        
        right_answer = solve(left, right + step_size)
        right_range = right_answer[1] - right_answer[0]
        
        both_answer = solve(left - step_size, right + step_size)
        both_range = both_answer[1] - both_answer[0]
        
        min_window = (-2, 2)
        min_range = min_window[1] - min_window[0]

        if left_range < min_range:
            min_range = left_range
            min_window = left_answer
            
        if right_range < min_range:
            min_range = right_range
            min_window = right_answer
            
        if both_range < min_range:
            min_range = both_range
            min_window = both_answer
        
        return min_window
        
    left, right = solve(1, 1)
    return left, right

def calculate_condition_indices(X, intercept_col_name):
    copy_X = X.copy()
    if intercept_col_name in copy_X: copy_X.drop(intercept_col_name, axis=COL, inplace=True)
    
    X_values = copy_X.values
    eigenvalues = np.linalg.eigvals(X_values.T @ X_values)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    
    condition_indices = np.sqrt(eigenvalues_sorted[0] / eigenvalues_sorted)
    condition_indices_rounded = ["{:.4f}".format(idx) for idx in condition_indices]
    
    condition_indices_df = pd.DataFrame({
        'Column Name': X.columns[sorted_indices],
        'Condition Index': condition_indices_rounded
    })
    return condition_indices_df

def get_model_error_assess_df(predicted_y, actual_y):
    results_df = pd.DataFrame()

    results_df['predicted_y'] = np.array(predicted_y)
    results_df['actual_y'] = np.array(actual_y)
    results_df['residuals'] = results_df['predicted_y'] - results_df['actual_y']
    
    return results_df

def get_beta_varcov_matrix(X_train, residuals, intercept_col_name, regularized_lambda = 0):
    copy_X = X_train.copy()
    # if intercept_col_name in copy_X: copy_X.drop(intercept_col_name, axis = 1, inplace=True)
    
    XtX = np.dot(copy_X.T, copy_X)
    XtX_adjusted_inv = np.linalg.inv(XtX + regularized_lambda * np.eye(copy_X.shape[1]))
    residual_variance = np.var(residuals, ddof=copy_X.shape[1])

    beta_variance = residual_variance * ( XtX_adjusted_inv @ XtX @ XtX_adjusted_inv)
    return beta_variance

def inverse_std_intercept(model, X_train, y_train, intercept_col_name):
    copy_X = X_train.copy()
    if intercept_col_name in copy_X: copy_X.drop(intercept_col_name, axis = 1, inplace=True)
    
    std_responsePredictors_ratio = (y_train.std() / copy_X.std())
    model_coef, model_intercept = model.coef_[:-1], model.coef_[-1]
    std_y_x_ratio = y_train.std() / copy_X.std()
    
    return model_intercept * y_train.std() + y_train.mean() - np.sum(model_coef * std_y_x_ratio * copy_X.mean())

def inverse_std_coef(model, X_train, y_train, intercept_col_name):
    copy_X = X_train.copy()
    if intercept_col_name in copy_X: copy_X.drop(intercept_col_name, axis = 1, inplace=True)
    
    std_responsePredictors_ratio = (y_train.std() / copy_X.std())
    model_coef = model.coef_[:-1]
    
    return model_coef * std_responsePredictors_ratio

def get_outliers_influential_points(X, y):
    from scipy.stats import t
    # Calculate the hat matrix
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    
    # Calculate the residuals
    y_hat = H @ y
    residuals = y - y_hat
    
    # Calculate the standard deviation of the residuals
    residual_std = np.sqrt(np.sum(residuals**2) / (X.shape[0] - X.shape[1]))
    
    # Calculate the standard errors
    standard_errors = np.sqrt(np.diag((1 - np.diag(H)) * residual_std**2))
    
    # Calculate the studentized residuals
    studentized_residuals = residuals / standard_errors
    
    # Identify outliers (two-sided test for absolute residuals)
    outliers = np.abs(studentized_residuals) > t.ppf(0.975, df=X.shape[0]-X.shape[1])
    
    # Identify influential points (Cook's distance)
    cooks_distance = residuals**2 / (X.shape[1] * standard_errors**2)
    influential_points = cooks_distance > 4 / (X.shape[0] - X.shape[1])
    
    # Identify leverage points (diagonal elements of the hat matrix)
    leverage_points = np.diag(H) > 2 * X.shape[1] / X.shape[0]
    
    # Print the indices of the outliers, influential points, and leverage points
    print("Outliers: ", np.where(outliers)[0])
    print("Influential points: ", np.where(influential_points)[0])
    print("Leverage points: ", np.where(leverage_points)[0])
    
    return outliers, influential_points, leverage_points

def get_standardize_df(X, intercept_col_name):
    X_copy = X.copy()
    
    non_encoding_colnames = [col for col in X_copy.columns if \
                            X_copy[col].dtype != 'int64' or X_copy[col].nunique() >= 10]
    
    mean_sub_df = X_copy[non_encoding_colnames].mean()
    std_sub_df = X_copy[non_encoding_colnames].std()

    X_copy[non_encoding_colnames] = (X_copy[non_encoding_colnames] - mean_sub_df) / std_sub_df
    X_copy[intercept_col_name] = 0
    
    return X_copy

def detect_outliers_leverage_influential(X_train, y_train):
    copy_X = X_train.copy()
    if 'const' not in copy_X: sm.add_constant(copy_X)
    
    model = sm.OLS(y_train, copy_X)
    results = model.fit()

    # Leverage is calculated as the diagonal of the hat matrix
    influence = results.get_influence()
    leverage = influence.hat_matrix_diag

    # Standardized residuals are used to find outliers
    standardized_residuals = results.resid_pearson

    # Cook's distance is a measure of the influence of each observation
    cooks_d = influence.cooks_distance[0]

    # You can adjust these thresholds as needed
    outlier_threshold = 2
    leverage_threshold = 2*copy_X.shape[1]/copy_X.shape[0]
    cooks_d_threshold = 4/copy_X.shape[0]

    outlier_indices = copy_X.index[np.abs(standardized_residuals) > outlier_threshold]
    high_leverage_indices = copy_X.index[leverage > leverage_threshold]
    influential_point_indices = copy_X.index[cooks_d > cooks_d_threshold]

    return outlier_indices, high_leverage_indices, influential_point_indices

def get_boxcox(y):
    from scipy import stats
    boxcox_y, lambda_val = stats.boxcox(y)
    return boxcox_y, lambda_val

def get_inv_boxcox(y, lambda_val):
    from scipy.special import inv_boxcox
    return inv_boxcox(y, lambda_val)


PATH = '/Users/hoangchu/Downloads/MATH158_Final/Data/cleaned_data.csv'
RESPONSE_COL_NAME = 'price'
ROW = 0
COL = 1


df = pd.read_csv(PATH)


y = df[RESPONSE_COL_NAME]
X = df.drop(RESPONSE_COL_NAME, axis = COL)
X = sm.add_constant(X)


print(X.shape)
print(X.dtypes.unique())


from sklearn.model_selection import train_test_split

# If you don’t specify the random_state in your code, then every time you run your code 
# a new random value is generated and the train and test datasets would have different values each time.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
assert y_train.min() <= y_test.min() <= y_test.max() <= y_train.max()


plot_freq_distribution(y_train, 2000, 'orange')
plot_freq_distribution(y_test, 500, 'blue')


# # Full Model


full_linear_model = sm.OLS(y_train, X_train).fit()
full_linear_model.summary()


full_predicted_y_train = full_linear_model.predict(X_train)
full_predicted_y_test = full_linear_model.predict(X_test)


full_results_df = get_model_error_assess_df(full_predicted_y_test, y_test)
full_results_df['weighted_residuals'] = full_results_df['residuals'].apply(lambda x: 1.2 * x if x < 0 else 0.8 * x)
print(np.mean(np.abs(full_results_df['weighted_residuals'])))


plot_qq(full_results_df['residuals'])
plot_residual_vs_predicted(full_results_df.residuals, full_results_df['predicted_y'])


X_train_corr_df = X_train.corr()


fully_connected_col_pairs = get_fully_correlated_pairs_set(X_train_corr_df)
for fully_connected_col_pair in fully_connected_col_pairs:
    col1, col2 = fully_connected_col_pair
    if col1 in X_train.columns: X_train.drop(col1, inplace=True, axis=COL)
print(fully_connected_col_pairs)


X_train_corrs = get_correlations(X_train_corr_df, 'const')
plot_freq_distribution(X_train_corrs, 50, 'orange')




std_X_train = get_standardize_df(X_train, 'const')
std_y_train = (y_train - y_train.mean()) / y_train.std()


# # LASSO AIC


from sklearn.linear_model import LassoLarsIC
aic_lasso_model = LassoLarsIC(criterion="bic", positive=True, fit_intercept=False).fit(std_X_train, std_y_train)


lassoAIC_stayed_columns = list(X_train.columns[aic_lasso_model.coef_ != 0])
if 'const' not in lassoAIC_stayed_columns: lassoAIC_stayed_columns.append('const')

lassoAIC_X_train = X_train[lassoAIC_stayed_columns]
lassoAIC_X_test = X_test[lassoAIC_stayed_columns]

print(len(lassoAIC_stayed_columns))
lassoAIC_linear_model = sm.OLS(y_train, lassoAIC_X_train).fit()
lassoAIC_linear_model.summary()


lassoAIC_predicted_y_train = lassoAIC_linear_model.predict(lassoAIC_X_train)
lassoAIC_predicted_y_test = lassoAIC_linear_model.predict(lassoAIC_X_test)

lassoAIC_train_results_df = get_model_error_assess_df(predicted_y=lassoAIC_predicted_y_train, actual_y=y_train)
lassoAIC_train_results_df['weighted_residuals'] = lassoAIC_train_results_df['residuals'].apply(lambda x: 1.2 * x if x < 0 else 0.8 * x)
print(np.mean(np.abs(lassoAIC_train_results_df['weighted_residuals'])))

lassoAIC_results_df = get_model_error_assess_df(predicted_y=lassoAIC_predicted_y_test, actual_y=y_test)
lassoAIC_results_df['weighted_residuals'] = lassoAIC_results_df['residuals'].apply(lambda x: 1.2 * x if x < 0 else 0.8 * x)
print(np.mean(np.abs(lassoAIC_results_df['weighted_residuals'])))

lassoAIC_beta_varcov_matrix = get_beta_varcov_matrix(lassoAIC_X_train, lassoAIC_results_df.residuals, 'const')
print(np.mean(np.diag(lassoAIC_beta_varcov_matrix)))

plot_qq(lassoAIC_results_df['residuals'])
plot_residual_vs_predicted(lassoAIC_results_df.residuals, lassoAIC_results_df['predicted_y'])


# # Ridge


# https://stats.stackexchange.com/questions/74622/converting-standardized-betas-back-to-original-variables


lamdas = np.arange(0.1, 100, 1.1)
lassoAIC_vif_df, ridge_beta_df = vif_custom(lassoAIC_X_train, lamdas, 'const')

left, right = find_optimal_ridge_alpha(lassoAIC_vif_df, .01)
lower_bound_ridge_condition = (lassoAIC_vif_df.min(axis=COL) >= left)
upper_bound_ridge_condition = (lassoAIC_vif_df.max(axis=COL) <= right)
lassoAIC_ridge_alpha = lassoAIC_vif_df[lower_bound_ridge_condition & upper_bound_ridge_condition].index[0]
print(f"lassoAIC_ridge_alpha = {lassoAIC_ridge_alpha}")

std_lassoAIC_X_train = get_standardize_df(lassoAIC_X_train, 'const')

from sklearn.linear_model import Ridge
lassoAIC_ridge_model = Ridge(alpha=lassoAIC_ridge_alpha, fit_intercept = False).fit(std_lassoAIC_X_train, std_y_train)

estimated_intercept = inverse_std_intercept(lassoAIC_ridge_model, lassoAIC_X_train, y_train, 'const')
esimated_coef = inverse_std_coef(lassoAIC_ridge_model, lassoAIC_X_train, y_train, 'const')


ridge_predicted_y_train = (lassoAIC_X_train.drop('const', axis=1) @ esimated_coef) + estimated_intercept
ridge_predicted_y_test = (lassoAIC_X_test.drop('const', axis=1) @ esimated_coef) + estimated_intercept

ridge_results_df = get_model_error_assess_df(predicted_y=ridge_predicted_y_test, actual_y=y_test)
ridge_results_df['weighted_residuals'] = ridge_results_df['residuals'].apply(lambda x: 1.2 * x if x < 0 else 0.8 * x)
print(np.mean(np.abs(ridge_results_df['weighted_residuals'])))

ridge_results_train_df = get_model_error_assess_df(predicted_y=ridge_predicted_y_train, actual_y=y_train)
ridge_results_train_df['weighted_residuals'] = ridge_results_train_df['residuals'].apply(lambda x: 1.2 * x if x < 0 else 0.8 * x)
print(np.mean(np.abs(ridge_results_train_df['weighted_residuals'])))

ridge_beta_varcov_matrix = get_beta_varcov_matrix(std_lassoAIC_X_train, ridge_results_df.residuals, 'const', lassoAIC_ridge_alpha)
print(np.mean(np.diag(ridge_beta_varcov_matrix)))

plot_qq(ridge_results_train_df['residuals'])
plot_residual_vs_predicted(ridge_results_train_df.residuals, ridge_results_train_df['predicted_y'])


# # Influential


outliers, leverage_points, influential_points = detect_outliers_leverage_influential(lassoAIC_X_train, y_train)

non_influential_outliers = outliers.intersection(influential_points)
print(len(non_influential_outliers) * 100 / len(outliers))

non_influential_leverage_points = leverage_points.intersection(influential_points)
print(len(non_influential_leverage_points) * 100 / len(leverage_points))

temp_y_train = y_train.drop(outliers, inplace=False)
temp_X_train = lassoAIC_X_train.drop(outliers, axis = ROW, inplace=False)

std_temp_X_train = get_standardize_df(temp_X_train, 'const')
std_temp_y_train = (temp_y_train - temp_y_train.mean()) / temp_y_train.std()

######
# bx_y, lambda_val = get_boxcox(temp_y_train)
# print(plot_freq_distribution(bx_y, 50, 'orange'))

# lassoAIC_linear_model = sm.OLS(bx_y, temp_X_train).fit()
# lassoAIC_linear_model.summary()

# lassoAIC_predicted_y_train = lassoAIC_linear_model.predict(temp_X_train)
# lassoAIC_predicted_y_test = lassoAIC_linear_model.predict(lassoAIC_X_test)

# lassoAIC_train_results_df = get_model_error_assess_df(predicted_y=lassoAIC_predicted_y_train, actual_y=temp_y_train)
# lassoAIC_train_results_df['weighted_residuals'] = lassoAIC_train_results_df['residuals'].apply(lambda x: 1.2 * x if x < 0 else 0.8 * x)
# print(np.mean(np.abs(lassoAIC_train_results_df['weighted_residuals'])))

# lassoAIC_results_df = get_model_error_assess_df(predicted_y=lassoAIC_predicted_y_test, actual_y=y_test)
# lassoAIC_results_df['weighted_residuals'] = lassoAIC_results_df['residuals'].apply(lambda x: 1.2 * x if x < 0 else 0.8 * x)
# print(np.mean(np.abs(lassoAIC_results_df['weighted_residuals'])))

# lassoAIC_beta_varcov_matrix = get_beta_varcov_matrix(temp_X_train, lassoAIC_results_df["residuals"], 'const')
# print(np.mean(np.diag(lassoAIC_beta_varcov_matrix)))

# plot_qq(lassoAIC_results_df['residuals'])
# plot_residual_vs_predicted(lassoAIC_results_df.residuals, lassoAIC_results_df['predicted_y'])
######

temp_predicted_y_train = (temp_X_train.drop('const', axis=1) @ esimated_coef) + estimated_intercept
temp_predicted_y_test = (lassoAIC_X_test.drop('const', axis=1) @ esimated_coef) + estimated_intercept
std_temp_X_train = get_standardize_df(temp_X_train, 'const')

temp_train_results_df = get_model_error_assess_df(predicted_y=temp_predicted_y_train, actual_y=temp_y_train)
temp_train_results_df['weighted_residuals'] = temp_train_results_df['residuals'].apply(lambda x: 1.2 * x if x < 0 else 0.8 * x)
print(np.mean(np.abs(temp_train_results_df['weighted_residuals'])))

temp_results_df = get_model_error_assess_df(predicted_y=temp_predicted_y_test, actual_y=y_test)
temp_results_df['weighted_residuals'] = temp_results_df['residuals'].apply(lambda x: 1.2 * x if x < 0 else 0.8 * x)
print(np.mean(np.abs(temp_results_df['weighted_residuals'])))

temp_beta_varcov_matrix = get_beta_varcov_matrix(std_temp_X_train, temp_results_df.residuals, 'const', lassoAIC_ridge_alpha)
print(np.mean(np.diag(temp_beta_varcov_matrix)))

plot_qq(temp_results_df['residuals'])
plot_residual_vs_predicted(temp_results_df.residuals, temp_results_df['predicted_y'])


np.sum(np.sum(lassoAIC_X_train.drop('const', axis=1).corr() - temp_X_train.drop('const', axis=1).corr(), axis=ROW))


# # LASSO


# from sklearn.linear_model import LassoCV
# lasso_model = LassoCV(alphas=None, cv = 10, fit_intercept = False, max_iter=2000).fit(minMax_X_train, minMax_y_train)

# lasso_stayed_columns = list(X_train.columns[lasso_model.coef_ != 0])
# if 'const' not in lasso_stayed_columns: lasso_stayed_columns.append('const')

# lasso_X_train = X_train[lasso_stayed_columns]
# lasso_X_test = X_test[lasso_stayed_columns]

# print(len(lasso_stayed_columns))
# lasso_linear_model = sm.OLS(y_train, lasso_X_train).fit()

# lasso_predicted_y_train = lasso_linear_model.predict(lasso_X_train)
# lasso_predicted_y_test = lasso_linear_model.predict(lasso_X_test)

# lasso_results_df = get_model_error_assess_df(predicted_y=lasso_predicted_y_test, actual_y=y_test)
# lasso_results_df['weighted_residuals'] = lasso_results_df['residuals'].apply(lambda x: 1.2 * x if x < 0 else 0.8 * x)
# print(np.mean(np.abs(lassoAIC_results_df['weighted_residuals'])))

# lasso_beta_varcov_matrix = get_beta_varcov_matrix(lasso_X_train, lasso_results_df.residuals, 'const')
# print(np.mean(np.diag(lasso_beta_varcov_matrix)))


