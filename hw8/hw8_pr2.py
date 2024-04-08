import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import reuters
from sklearn.feature_extraction import text
import time

#########################################
#			 Helper Functions	    	#
#########################################
def nmf_cost(X, W, H):
	"""	This function takes in three arguments:
			1) X, the data matrix with dimension m x n
			2) W, a matrix with dimension m x k
			3) H, a matrix with dimension k x n

		This function calculates and returns the cost defined by
		|X - WH|^2.
	"""
	# TODO: Calculate the cost for nmf algorithm
	cost = 0.

	sparse_X = X.tocoo()
	for index in range(len(sparse_X.data)):
		row = sparse_X.row[index]
		col = sparse_X.col[index]
		data = sparse_X.data[index]
		cost += (data - (W[row, :] @ H[:, col]).item(0)) ** 2

	return cost

def nmf(X, k=20, max_iter=100, print_freq=10):
	"""	This function takes in three arguments:
			1) X, the data matrix with dimension m x n
			2) k, the number of latent factors
			3) max_iter, the maximum number of iterations
			4) print_freq, the frequency of printing the report

		This function runs the nmf algorithm and returns the following:
			1) W, a matrix with dimension m x k
			2) H, a matrix with dimension k x n
			3) cost_list, a list of costs at each iteration
	"""
	m, n = X.shape
	W = np.abs(np.random.randn(m, k) * 1e-3)
	H = np.abs(np.random.randn(k, n) * 1e-3)
	cost_list = [nmf_cost(X, W, H)]
	t_start = time.time()
	
	for iter_num in range(max_iter):

		H = H * (W.T @ X) / ((W.T @ W) @ H)
		W = W * (X @ H.T) / (W @ (H @ H.T))
		cost = nmf_cost(X, W, H)
		cost_list.append(cost)

		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - cost: {:.4E}'.format(iter_num + 1, \
				cost))

	# Benchmark report
	t_end = time.time()
	print('--Time elapsed for running nmf: {t:4.2f} seconds'.format(\
			t=t_end - t_start))

	return W, H, cost_list

###########################################
#	    	Main Driver Function       	  #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	print('==> Loading data...')

	X = np.array([' '.join(list(reuters.words(file_id))).lower() \
		for file_id in reuters.fileids()])
	tfidf = text.TfidfVectorizer()
	X = tfidf.fit_transform(X)
	# =============STEP 1: RUNNING NMF=================

	print('==> Running nmf algorithm on the dataset...')
	W, H, cost_list = nmf(X, k=20, print_freq=10)
	# =============STEP 2: CONVERGENCE PLOT=================
	print('==> Generating convergence plot...')
	plt.style.use('ggplot')
	plt.plot(cost_list)
	plt.xlabel('iteration')
	plt.ylabel('cost')
	plt.title('Reuters NMF Convergence Plot with {} Topics'.format(H.shape[0]))
	plt.savefig('nmf_cvg.png', format='png')
	plt.close()
	# =============STEP 3: FIND MOST FREQUENT WORDS=================
	print('==> Finding most frequent words for each topic...')
	num_top_words = 10


	ind = np.flip(np.argsort(H, axis=1), 1)
	top_words = np.array(tfidf.get_feature_names())[ind]
	np.set_printoptions(threshold=np.nan)
	for topic_ind in range(H.shape[0]):
		print('-- topic {}: {}'.format(topic_ind + 1, top_words[topic_ind, :num_top_words]))
