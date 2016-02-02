import sys
import numpy as np
import matplotlib
matplotlib.use('Gtk3Agg')
import matplotlib.pyplot as plt
np.random.seed(None)

class DataGen:

	def __init__(self, bias, data_dim=2):
		self.coeffs = np.random.rand(data_dim+1)
		for i in range(1, data_dim+1):
			self.coeffs[i] /= np.sum(self.coeffs[1:]) / (2 * self.coeffs[0])

		self.bias = bias
		self.data_dim = data_dim
	
	def data_gen(self, num_cases=40): # generate linearly separable data, roughly 50% of each class
		data = np.ones((int(num_cases), self.data_dim+1))
		for i in range(num_cases):
			data_tmp = np.random.rand(self.data_dim)
			while ((0.5 - self.coeffs[0]) == data_tmp.dot(self.coeffs[1:])):
				data_tmp = np.random.rand(self.data_dim)
			data[i] = np.insert(data_tmp, data_tmp.size, self.coeffs[0] < data_tmp.dot(self.coeffs[1:]))
		data = np.insert(data, 0, self.bias, axis=1)
		return data

class Perceptron:
	
	def __init__(self, learn_rate=0.01, data_dim=2, train_set_size=40):
		self.bias = 1.0
		self.rate = learn_rate
		self.DG = DataGen(self.bias, data_dim)
		self.training_data = self.DG.data_gen(train_set_size)
		self.weights = np.zeros((self.training_data[0].size-1))
		self.run()

	def step_func(self, step_input):
		if (step_input <= 0.5):
			return 0
		else:
			return 1
	
	def train(self):
		err_count = 0
		for i in range(self.training_data[:,0].size): # number of training cases
			perceptron_label = self.step_func(self.weights.dot(self.training_data[i, :self.weights.size]))
			error = self.training_data[i, self.weights.size] - perceptron_label
			if (error != 0):
				err_count += 1
			for j in range(self.weights.size):
				self.weights[j] += error * self.rate * self.training_data[i, j]
		# print(self.weights)
		return err_count

	def run(self):
		count = 0
		while True:
			count += 1
			err = self.train()
			if (err == 0):
				break
		print('Training complete')
		print('Gen ' + str(count) + ' : wgts ' + str(self.weights))
		print('Ground Truth: ' + str(self.DG.coeffs))
	
	def test(self, test_cases=120):
		err_count = 0
		test_data = self.DG.data_gen(test_cases)
		for i in range(test_data[:,0].size):
			tmp = test_data[i, 0:self.weights.size].dot(self.weights)
			perceptron_label = self.step_func(tmp)
			error = test_data[i, self.weights.size] - perceptron_label
			if (error != 0):
				err_count += 1
		print('Testing complete: err ' + str(err_count / test_cases * 100) + '%')
		return test_data

def plot_data_2D(perceptron, test_cases=120): # ONLY WORKS WITH 2D DATA
	p = perceptron
	test_data = p.test(test_cases)

	training_data_0, training_data_1 = [], []
	for i in range(p.training_data[:, 0].size):
		if (p.training_data[i, p.weights.size] == 0):
			training_data_0.append(p.training_data[i, 1:3])
		else:
			training_data_1.append(p.training_data[i, 1:3])
	training_data_0 = np.asmatrix(training_data_0)
	training_data_1 = np.asmatrix(training_data_1)

	test_data_0, test_data_1 = [], []
	for i in range(test_data[:, 0].size):
		if (test_data[i, p.weights.size] == 0):
			test_data_0.append(test_data[i, 1:3])
		else:
			test_data_1.append(test_data[i, 1:3])
	test_data_0 = np.asmatrix(test_data_0)
	test_data_1 = np.asmatrix(test_data_1)

	line_truth_x = np.linspace(0, 1, num=6, endpoint=True)
	line_truth_y = (-p.DG.coeffs[1] * line_truth_x + p.DG.coeffs[0]) / p.DG.coeffs[2]
	line_percept_x = np.linspace(0, 1, num=6, endpoint=True)
	# 0.5 - weights[0] because range is 0->1
	line_percept_y = (-p.weights[1] * line_percept_x + (0.5 - p.weights[0])) / p.weights[2]

	h0, = plt.plot(training_data_0[:, 0], training_data_0[:, 1], 'bo', label='training0')
	h1, = plt.plot(training_data_1[:, 0], training_data_1[:, 1], 'ro', label='training1')
	h2, = plt.plot(test_data_0[:, 0], test_data_0[:, 1], 'co', label='test0')
	h3, = plt.plot(test_data_1[:, 0], test_data_1[:, 1], 'mo', label='test1')
	h4, = plt.plot(line_truth_x, line_truth_y, color='k', linewidth=1, label='ground truth')
	h5, = plt.plot(line_percept_x, line_percept_y, color='g', linewidth=1, label='decision boundary')
	plt.axis([0, 1, 0, 1])
	#plt.legend(handles=[h0, h1, h2, h3, h4, h5], ncol=2, loc=2, borderaxespad=0)
	print('\nLEGEND:')
	print('[blue dots, training_data_0] [red dots,     training_data_1]')
	print('[cyan dots,     test_data_0] [magenta dots,     test_data_1]')
	print('[black line,      true_line] [green line, decision_boundary]')
	plt.show()

def plot_data_3D(perceptron, test_cases=120):
	p = perceptron
	test_data = p.test(test_cases)
	print('\n## 3D plotting not yet implemented ##')

print('\nUsage listed below in following format: ARG(=default_value)')
print('Usage: python perceptron.py LEARN_RATE(=0.01) DATA_DIM(=2) TRAINING_CASES(=40) TEST_CASES(=120)\n')
if (len(sys.argv) != 6):
	p = Perceptron()
	plot_data_2D(p)
else:
	p = Perceptron(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))
	if (int(sys.argv[3]) == 2):
		plot_data_2D(p, int(sys.argv[4]))
	elif (int(sys.argv[3]) == 3):
		plot_data_3D(p, int(sys.argv[4]))