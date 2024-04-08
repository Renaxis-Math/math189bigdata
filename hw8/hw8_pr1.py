import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from sklearn.metrics.pairwise import rbf_kernel
#########################################
#			 Helper Functions	    	#
#########################################

def k_means(X, k, eps=1e-10, max_iter=1000, print_freq=10):
	"""	This function takes in the following arguments:
			1) X, the data matrix with dimension m x n
			2) k, the number of clusters
			3) eps, the threshold of the norm of the change in clusters
			4) max_iter, the maximum number of iterations
			5) print_freq, the frequency of printing the report

		This function returns the following:
			1) clusters, a list of clusters with dimension k x 1
			2) label, the label of cluster for each data with dimension m x 1
			3) cost_list, a list of costs at each iteration
	"""
	m, n = X.shape
	cost_list = []
	t_start = time.time()
	clusters = np.random.multivariate_normal((.5 + np.random.rand(n)) * X.mean(axis=0), 10 * X.std(axis=0) * np.eye(n), \
		size=k)
	label = np.zeros((m, 1)).astype(int)
	iter_num = 0

	while iter_num < max_iter:
		prev_clusters = copy.deepcopy(clusters)
		# find closets center for each data point
		for i in range(m):
			data = X[i, :]
			diff = data - clusters
			curr_label = np.argsort(np.linalg.norm(diff, axis=1)).item(0)
			label[i] = curr_label
		# update centers
		for cluster_num in range(k):
			ind = np.where(label == cluster_num)[0]
			if len(ind) > 0:
				clusters[cluster_num, :] = X[ind].mean(axis=0)
		# calculate costs
		cost = k_means_cost(X, clusters, label)
		cost_list.append(cost)
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - cost {:4.4E}'.format(iter_num + 1, cost))
		if np.linalg.norm(prev_clusters - clusters) <= eps:
			print('-- Algorithm converges at iteration {} \
				with cost {:4.4E}'.format(iter_num + 1, cost))
			break
		iter_num += 1

	t_end = time.time()
	print('-- Time elapsed: {t:2.2f} \
		seconds'.format(t=t_end - t_start))
	return clusters, label, cost_list

def k_means_cost(X, clusters, label):
	"""	This function takes in the following arguments:
			1) X, the data matrix with dimension m x n
			2) clusters, the matrix with dimension k x 1
			3) label, the label of the cluster for each data point with
				dimension m x 1

		This function calculates and returns the cost for the given data
		and clusters.
	"""
	m, n = X.shape
	k = clusters.shape[0]

	X_cluster = clusters[label.flatten()]
	cost = (np.linalg.norm(X - X_cluster, axis=1) ** 2).sum()

	return cost

###########################################
#	    	Main Driver Function       	  #
###########################################


if __name__ == '__main__':
	print('==> Step 0: Loading data...')
	# Read data
	path = '5000_points.csv'
	columns = ['x', 'space', 'y']
	features = ['x', 'y']
	df = pd.read_csv(path, sep='  ', names = columns, engine='python')
	X = np.array(df[:][features]).astype(int)

	print('==> Step 1: Finding optimal number of clusters...')
	cost_k_list = []
	for k in range(1, 21):
		"*** YOUR CODE HERE ***"
		clusters, label, cost_list = k_means(X, k)
		cost = cost_list[-1]
		cost_k_list.append(cost)

		print('-- Number of clusters: {} - cost: {:.4E}'.format(k, cost))
	opt_k = np.argmin(cost_k_list) + 1
	print('-- Optimal number of clusters is {}'.format(opt_k))

	"*** YOUR CODE HERE ***"
	cost_vs_k_plot, = plt.plot(range(1, 21), cost_k_list, 'g^')
	opt_cost_plot, = plt.plot(opt_k, min(cost_k_list), 'rD')
	"*** END YOUR CODE HERE ***"
	plt.title('Cost vs Number of Clusters')
	plt.savefig('kmeans_1.png', format='png')
	plt.close()

	"*** YOUR CODE HERE ***"
	clusters, label, cost_list = k_means(X, opt_k)
	X_cluster = clusters[label.flatten()]
	data_plot, = plt.plot(X[:, 0], X[:, 1], 'bo')
	center_plot, = plt.plot(X_cluster[:, 0], X_cluster[:, 1], 'rD')
	"*** END YOUR CODE HERE ***"
	# set up legend and save the plot to the current folder
	plt.legend((data_plot, center_plot), \
		('data', 'clusters'), loc = 'best')
	plt.title('Visualization with {} clusters'.format(opt_k))
	plt.savefig('kmeans_2.png', format='png')
	plt.close()