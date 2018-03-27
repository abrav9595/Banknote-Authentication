import numpy, math
import random
import matplotlib.pyplot as plot

def get_Training_and_Testing_set_indexes(iteration_number, percentage_of_training_data):
	if iteration_number is 1:
		training_set_indices, test_set_indices = range(0, int(data.shape[0]*0.66667)), range(int(data.shape[0]*0.66667)+1, data.shape[0])
	elif iteration_number is 2:
		training_set_indices, test_set_indices = range(0, int(data.shape[0]*0.33334)) + range(int(data.shape[0]*0.66667)+1, data.shape[0]), range(int(data.shape[0]*0.33334)+1, int(data.shape[0]*0.66667))
	else:
		training_set_indices, test_set_indices = range(int(data.shape[0]*0.33334)+1, data.shape[0]), range(0, int(data.shape[0]*0.33334))

	return numpy.random.choice(numpy.asarray(training_set_indices), int(len(training_set_indices)*percentage_of_training_data), False).tolist(), test_set_indices

def sigmoid_function(features, weights):
	return 1.0/(1+numpy.exp(-numpy.dot(features, weights.T)))


def find_accuracy(model_predictions, output_Y):
	for i in range(0, model_predictions.shape[0]):
		if model_predictions[i][0] >= 0.5:
			model_predictions[i][0] = 1
		else:
			model_predictions[i][0] = 0

	return 1.0 - ((numpy.absolute(output_Y - model_predictions)).sum()/model_predictions.shape[0])

def find_mean_variance_probofoutputs(training_set):
	training_set_positive_naive_bayes_indices = []
	for i in range(0,training_set.shape[0]):
		if training_set[i][5] in (1.0,1):
			training_set_positive_naive_bayes_indices.append(i)
	training_set_negative_naive_bayes_indices = list(set(range(0,training_set.shape[0])) - set(training_set_positive_naive_bayes_indices))

	mean_positive_examples, variance_positive_examples = numpy.mean(training_set[training_set_positive_naive_bayes_indices,1:5], axis=0).tolist(), numpy.var(training_set[training_set_positive_naive_bayes_indices,1:5], axis=0).tolist()
	mean_negative_examples, variance_negative_examples = numpy.mean(training_set[training_set_negative_naive_bayes_indices,1:5], axis=0).tolist(), numpy.var(training_set[training_set_negative_naive_bayes_indices,1:5], axis=0).tolist()

	probability_of_success = (training_set[:,5:6]).sum()/(training_set[:,5:6]).shape[0]
	
	return (mean_positive_examples, variance_positive_examples, mean_negative_examples, variance_negative_examples, probability_of_success)

def getPredictions_NB(test_features_X, naive_bayes_model):
	predictions = []
	if naive_bayes_model is None:
		# Random prediction if there is not training Data
		predictions = [1 if random.random() >=0.5 else 0 for i in range(0,test_features_X.shape[0])]
	else:
		for i in range(0,test_features_X.shape[0]):
			likelihood_pos, likelihood_neg = 1.0, 1.0
			probability_of_success = naive_bayes_model[4]
			if probability_of_success is 0.0:
				likelihood_pos = 0.0
			elif probability_of_success is 1.0:
				likelihood_neg = 0.0
		
			# calculating likelihood for each feature
			for j in range(0,test_features_X.shape[1]):
				try:
					mean_pos, var_pos, mean_neg, var_neg = naive_bayes_model[0][j], naive_bayes_model[1][j], naive_bayes_model[2][j], naive_bayes_model[3][j]
					if int(var_pos) is 0 or int(var_neg) is 0:
						continue
					if likelihood_pos is not 0.0:
						likelihood_pos = likelihood_pos * ((1/math.sqrt(2*math.pi*var_pos)) * math.exp((-1 * math.pow(test_features_X[i][j] - mean_pos, 2)) / (2 * var_pos)))
					if likelihood_neg is not 0.0:
						likelihood_neg = likelihood_neg * ((1/math.sqrt(2*math.pi*var_neg)) * math.exp((-1 * math.pow(test_features_X[i][j] - mean_neg, 2)) / (2 * var_neg)))
				except Exception as e:
					print(e)
					print naive_bayes_model
					input("temp")

			probability_of_predicting_success = (probability_of_success * likelihood_pos) / ((probability_of_success * likelihood_pos) + ((1 - probability_of_success) * likelihood_neg))

			if probability_of_predicting_success >= 0.5:
				predictions.append(1.0)
			else:
				predictions.append(0.0)
	return predictions

def train(percentage_of_training_data, learning_rate):
	# Initializing weight vector
	weights = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0])[numpy.newaxis] # 1x4 vector as we have 4 attributes
	total_accuracy_regression, total_accuracy_NB = 0.0, 0.0
	
	for iteration_number in range(1,4):
		
		# Get bounds of training set and test set for this iteration of cross validation
		training_set_indices, test_set_indices = get_Training_and_Testing_set_indexes(iteration_number, percentage_of_training_data)

		# Training set and Test set
		training_set = data[training_set_indices,:]
		test_set = data[test_set_indices,:]

		# Training Features
		features_X = training_set[:,0:5]
		output_Y = training_set[:,5:6]

		# Test Features
		test_features_X = test_set[:,0:5]
		test_output_Y = test_set[:,5:6]

		# training begins for logistic regression
		for i in range(0,50000):
			model_predictions = sigmoid_function(features_X, weights) # Predicting the model with the current weights.

			# modifying weights by adding to it the gradient of the log likelihood w.r.t each weight.
			weights = weights + learning_rate*numpy.dot(features_X.T, output_Y - model_predictions).T

		# Using the test set to find the fitness of my regression model
		model_predictions = sigmoid_function(test_features_X, weights)

		# Training begins for naive bayes
		naive_bayes_model, model_predictions_NB = None, None
		if percentage_of_training_data not in (0.001, 0.005):
			naive_bayes_model = find_mean_variance_probofoutputs(training_set)

		# Using the test set to find the fitness of my naive bayes model
		model_predictions_NB = getPredictions_NB(test_features_X[:,1:5], naive_bayes_model)
		total_accuracy_NB = total_accuracy_NB + find_accuracy(numpy.asarray([[prediction] for prediction in model_predictions_NB]), test_output_Y)

		total_accuracy_regression = total_accuracy_regression + find_accuracy(model_predictions, test_output_Y)
	
	print "	Percentage "+str(percentage_of_training_data)
	print "		Regression Accuracy % = "+str(total_accuracy_regression/3.0)
	print "		Naive Bayes Accuracy % = "+str(total_accuracy_NB/3.0)
	return (total_accuracy_regression/3.0, total_accuracy_NB/3.0)

def compareNB():
	#Logic same as train function, where we get the indices for training set and test set and then obtain the naivebayesmodel
	for iteration_number in range(1,4):
		training_set_indices, test_set_indices = get_Training_and_Testing_set_indexes(iteration_number, 1)
		training_set = data[training_set_indices,:]
		naive_bayes_model = find_mean_variance_probofoutputs(training_set)

		print str(iteration_number)+" fold of cross validation:-"
		# Generate equal number of samples from each gaussian distribution P(X/Y)
		samples_features = None
		for i in range(0,4):
			mean_pos, var_pos = naive_bayes_model[0][i], naive_bayes_model[1][i]
			new_sample_feature = numpy.random.normal(mean_pos, math.sqrt(var_pos), size=(400,1))
			if samples_features is None:
				samples_features = new_sample_feature
			else:
				samples_features = numpy.concatenate((samples_features, new_sample_feature), axis=1)
		sample_mean, sample_var = numpy.mean(samples_features, axis = 0), numpy.var(samples_features, axis = 0)
		print "	Sample Mean "+str(iteration_number)+" = "+str(sample_mean)
		print "	Training Set Means "+str(iteration_number)+" = "+str(naive_bayes_model[0])
		print "	Sample Variance "+str(iteration_number)+" = "+str(sample_var)
		print "	Training Set Variance "+str(iteration_number)+" = "+str(naive_bayes_model[1])

if __name__ == "__main__":
	from numpy import genfromtxt

	# Get the data from CSV and store it in arra
	data = genfromtxt('data.csv', delimiter=',')

	temp = (data.shape[0],1)
	bias_feature = numpy.ones(temp)
	data = numpy.concatenate((bias_feature, data), axis=1)
	
	# Randomly Shuffling the data
	numpy.random.shuffle(data)
	
	# Train the model for 5 runs
	percentage_of_training_data = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.625, 1]
	average_accuracies_regression = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	average_accuracies_NB = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	for run in range(1,6):
		print "run "+str(run)
		accuracies_regression = []
		accuracies_NB = []
		for percentage in percentage_of_training_data:
			accuracies = train(percentage, 0.0001)
			accuracies_regression.append(accuracies[0])
			accuracies_NB.append(accuracies[1])
		average_accuracies_regression = (numpy.asarray(accuracies_regression) + numpy.asarray(average_accuracies_regression)).tolist()
		average_accuracies_NB = (numpy.asarray(accuracies_NB) + numpy.asarray(average_accuracies_NB)).tolist()
	
	#Averaging the accuracies over 5 runs
	average_accuracies_regression = (numpy.asarray(average_accuracies_regression)/5.0).tolist()
	average_accuracies_NB = (numpy.asarray(average_accuracies_NB)/5.0).tolist()

	# Compare Naive Bayes for last question 
	compareNB()

	# Plot the learning curve
	plot.plot(percentage_of_training_data, average_accuracies_regression, '-', label="Logistic Regression")
	plot.plot(percentage_of_training_data, average_accuracies_NB, '-', label="Naive Bayes")
	plot.xlabel('Percentage of Training Data')
	plot.ylabel('Accuracy %')
	plot.legend()
	plot.show()