import pandas as pd
import numpy as np

print('==>Loading data...')
name_list = ['pix_{}'.format(i + 1) for i in range(784)]
name_list = ['label'] + name_list
df_train = pd.read_csv('http://pjreddie.com/media/files/mnist_train.csv', \
	sep=',', engine='python', names = name_list)
df_test = pd.read_csv('http://pjreddie.com/media/files/mnist_test.csv', \
	sep=',', engine='python', names = name_list)
print('==>Data loaded succesfully.')
X_train = np.array(df_train[:][[col for col in df_train.columns \
if col != 'label']]) / 256.
y_train = np.array(df_train[:][['label']])
X_test = np.array(df_test[:][[col for col in df_test.columns \
if col != 'label']]) / 256.
y_test = np.array(df_test[:][['label']])