import numpy as np
import pandas as pd

class reg2HDFE(object):

	'''
	python class to fit a linear model with two high dimensional fixed effects
	according to the algorithm of Guimaraes and Portugal (2010)
	pandas and pure numpy implementations (for the groupby operations)
	numpy seems faster

	Sebastian Hohmann, 2019
	'''


	def __init__(self, data, xvec, y, i, t):

		self.M = data[y].shape[0]
		self.K = len(xvec)

		self.y = np.array(data[y]).reshape((self.M, 1))
		self.X = np.array(data[xvec])
		self.i = np.array(data[i]).reshape((self.M, 1))
		self.t = np.array(data[t]).reshape((self.M, 1))

	def fit_pandas(self):

		df = pd.DataFrame(np.concatenate([self.i, self.t, self.X], axis=1))
		# print(df.head())
		df.columns = ['i', 't'] + list(df.columns[2:])
		df_itcols = list(df.columns[2:]) + ['fe_i', 'fe_t']

		# print(df_itcols)

		rss2 = 0
		tol = 10e-12
		i = 0

		X = np.concatenate([self.X, np.ones((self.M, 1))], axis=1)
		beta_hat = np.linalg.inv((X.T).dot(X)).dot((X.T).dot(self.y))
		res = self.y - X.dot(beta_hat)
		rss1 = np.sum(res ** 2)
		dif = rss2-rss1
		df['res'] = res
		df['temp'] = res.reshape((self.M,))
		df['fe_i'] = df['temp'].groupby(df['i']).transform('mean')
		df['fe_t'] = df['temp'].groupby(df['t']).transform('mean')

		while abs(dif) > tol and i < 20:
			X = np.concatenate((np.array(df[df_itcols]), np.ones((self.M, 1))), axis=1)
			rss2 = rss1
			beta_hat = np.linalg.inv((X.T).dot(X)).dot((X.T).dot(self.y))
			res = self.y - X.dot(beta_hat)
			rss1 = np.sum(res ** 2)
			dif = rss2-rss1
			df['res'] = res    
			df['temp'] = res.reshape((self.M,)) + beta_hat[self.K][0]*df.fe_i
			df['fe_i'] = df['temp'].groupby(df['i']).transform('mean')
			df['temp'] = res.reshape((self.M,)) + beta_hat[self.K+1][0]*df.fe_t
			df['fe_t'] = df['temp'].groupby(df['t']).transform('mean')
			i+=1

			if (i+1) % 10 == 0:
				print(i+1, dif)

		self.yhat_pd = X.dot(beta_hat)

	def fit_numpy(self):

		groups_i, ids_i, counts_i = np.unique(self.i, return_inverse=True, return_counts=True)
		groups_t, ids_t, counts_t = np.unique(self.t, return_inverse=True, return_counts=True)

		rss2 = 0
		tol = 10e-12
		i = 0

		X = np.concatenate([self.X, np.ones((self.M, 1))], axis=1)
		beta_hat = np.linalg.inv((X.T).dot(X)).dot((X.T).dot(self.y))
		res = (self.y - X.dot(beta_hat)).flatten()
		rss1 = np.sum(res ** 2)
		dif = rss2-rss1

		fe_i = self.mean_bincount(res, ids_i, counts_i).reshape((self.M, 1))
		fe_t = self.mean_bincount(res, ids_t, counts_t).reshape((self.M, 1))

		while abs(dif) > tol and i < 20:
			X = np.concatenate([self.X, fe_i, fe_t, np.ones((self.M, 1))], axis=1)
			rss2 = rss1
			beta_hat = np.linalg.inv((X.T).dot(X)).dot((X.T).dot(self.y))
			res = (self.y - X.dot(beta_hat)).flatten()
			rss1 = np.sum(res ** 2)
			dif = rss2-rss1

			temp = res + beta_hat[self.K][0]*fe_i.flatten()
			fe_i = self.mean_bincount(temp, ids_i, counts_i).reshape((self.M, 1))
			temp = res + beta_hat[self.K][0]*fe_t.flatten()
			fe_t = self.mean_bincount(temp, ids_t, counts_t).reshape((self.M, 1))

			i+=1

			if (i+1) % 10 == 0:
				print(i+1, dif)

		self.yhat_np = X.dot(beta_hat)	


	def mean_bincount(self, vals, ids, counts):

		'''
		find mean of vector "vals" by group in "ids"
		pure numpy implementation
		https://stackoverflow.com/questions/29243982/average-using-grouping-value-in-another-vector-numpy-python
		'''

		means = np.bincount(ids, weights=vals) / counts
		return means[ids]


