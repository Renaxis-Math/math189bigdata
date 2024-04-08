import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#########################################
#			 Helper Functions	    	#
#########################################
def find_cost(X, y, W , reg):
	"""	This function takes in three arguments:
			1) W, a weight matrix with bias
			2) X, the data with dimension m x (n + 1)
			3) y, the label of the data with dimension m x 1

		This function calculates and returns the l1 regularized
		mean-squared error
	"""
	m = len(y)
	err = X @ W - y
	err = float(err.T @ err)
	return (err + reg * np.abs(W).sum()) / m

def find_grad(X, y, W, reg=0.0):
	"""	This function takes in four arguments:
			1) X, the data with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) W, a weight matrix with bias
			4) reg, the parameter for regularization

		This function calculates and returns the gradient of W
	"""
	m = X.shape[0]
	return X.T @ (X @ W - y) / m

def prox(X, gamma):
	""" This function takes in two arguments:
			1)  X, a vector
			2) gamma, a scalar

		This function thresholds each entry of X with gamma
		and updates the changes in place.
	"""
	X[np.abs(X) <= gamma] = 0.
	X[X > gamma] -= gamma
	X[X < -gamma] += gamma

	return X

def grad_lasso(
	X, y, reg=1e6, lr=1e-12, eps=1e-5,
	max_iter=300, batch_size=256, print_freq=1):
	""" This function takes in the following arguments:
			1) X, the data with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) reg, the parameter for regularization
			4) lr, the learning rate
			5) eps, the threshold of the norm for the gradients
			6) max_iter, the maximum number of iterations
			7) batch_size, the size of each batch for gradient descent
			8) print_freq, the frequency of printing the report

		This function returns W, the optimal weight,
		by lasso gradient descent.
	"""
	m, n = X.shape
	obj_list = []
	# initialize the weight and its gradient
	W = np.linalg.solve(X.T @ X, X.T @ y)
	W_grad = np.ones((n, 1))
	print('==> Running gradient descent...')
	iter_num = 0
	t_start = time.time()

	while iter_num < max_iter and np.linalg.norm(W_grad) > eps:
		# update W
		ind = np.random.randint(0, m, size=batch_size)
		W_grad = find_grad(X[ind], y[ind], W, reg=reg)
		# apply threshold function
		W = prox(W - lr * W_grad, reg * lr)
		# calculate MSE
		cost = find_cost(X[ind], y[ind], W, reg=reg)
		obj_list.append(cost)
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration{} - training cost {: .4f} - \
				sparsity {: .2f}'.format(iter_num + 1, cost, \
					(np.abs(W) < reg * lr).mean()))
		iter_num += 1

	# Benchmark report
	t_end = time.time()
	print('--Time elapsed for training: {t:4.2f} \
		seconds'.format(t = t_end - t_start))
	return W, obj_list

def lasso_path(X, y, tau_min=1e-8, tau_max=10, num_reg=10):
	""" This function takes in the following arguments:
			1) X, the data with dimension m x (n + 1)
			2) y, the label of the data with dimension m x 1
			3) tau_min, the minimum value for the inverse of regularization parameter
			4) tau_max, the maximum value for the inverse of regularization parameter
			5) num_reg, the number of regularization parameters

		This function returns the list of optimal weights and the corresponding tau values.
	"""
	m, n = X.shape
	W = np.zeros((n, num_reg))
	tau_list = np.linspace(tau_min, tau_max, num_reg)
	for index in range(num_reg):
		reg = 1. / tau_list[index]
		print('--regularization parameter is {:.4E}'.format(reg))
		W[:, index] = grad_lasso(X, y, reg=reg, lr=1e-12, \
			max_iter=1000, batch_size=1024, print_freq=1000)[0].flatten()
	return W, tau_list


###########################################
#	    	Main Driver Function       	  #
###########################################

if __name__ == '__main__':
	# =============STEP 0: LOADING DATA=================
	print('==> Step 0: Loading data...')
	# Read data
	df = pd.read_csv('https://math189r.github.io/hw/data/online_news_popularity/online_news_popularity.csv', \
		sep=', ', engine='python')
	X = df[[col for col in df.columns if col not in ['url', 'shares', 'cohort']]]
	y = np.log(df.shares).values.reshape(-1,1)
	X = np.hstack((np.ones_like(y), X))

	print('==> Step 2: Running lasso path...')
	W, tau_list = lasso_path(X, y, tau_min=1e-15, tau_max=2e-2, num_reg=10)
	# Plotting lasso path
	plt.style.use('ggplot')
	plt.subplot(211)
	lp_plot = plt.plot(tau_list, W.T)
	plt.title('Lasso Path')
	plt.xlabel('$tau = \lambda^{-1}$')
	plt.ylabel('$W_i$')

	print('==> Step 3: The most important features are: ')
	top_features = np.array(df.columns)[np.argsort(-W[:, 0])[:5] + 1]
	print(top_features)

	print('==> Step 4: Generating convergence plot...')
	plt.subplot(212)
	W_reg, obj_list = grad_lasso(X, y, reg=1e5, lr=1e-12, eps=1e-2, max_iter=2500, \
		batch_size=1024, print_freq=250)
	plt.title("Lasso Objective Convergence: $\lambda = 1e5$")
	plt.ylabel("Stochastic Objective")
	plt.xlabel("Iteration")
	plt.plot(obj_list)
	plt.tight_layout()
	plt.savefig('hw7pr2_lasso.png', format = 'png')
	plt.close()
	print('==> Plotting completed.')