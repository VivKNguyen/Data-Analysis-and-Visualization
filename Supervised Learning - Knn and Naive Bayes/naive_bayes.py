'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Vivian Nguyen
CS 251/2: Data Analysis Visualization
Spring 2023
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`.
        - Add placeholder instance variables the class prior probabilities and class likelihoods (assigned to None).
        You may store the priors and likelihoods themselves or the logs of them. Be sure to use variable names that make
        clear your choice of which version you are maintaining.
        '''
        self.num_classes = num_classes
        self.priors = None
        self.likelihoods = None
        pass

        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham

        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c

    def get_priors(self):
        '''Returns the class priors (or log of class priors if storing that)'''
        return self.priors
        pass

    def get_likelihoods(self):
        '''Returns the class likelihoods (or log of class likelihoods if storing that)'''
        return self.likelihoods
        pass

    def get_num_classes(self):
        return self.num_classes
    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the class priors and class likelihoods (i.e. your instance variables) that are needed for
        Bayes Rule. See equations in notebook.
        '''
        # y has 'num_samps' many classes
        # num_classes tells how many unique classes there are
        self.num_classes = len(np.unique(y))

        num_samples, num_features = data.shape

        # create empty array that is the shape of how many classes there are 
        class_counts = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            # count how many time a class appears
            class_counts[i] = np.sum(y == i)

        # Compute self.priors
        self.priors = class_counts / num_samples

        # create empty array that is the shape of what likelihoods should be
        self.likelihoods = np.zeros((self.num_classes, num_features))
        
        for i in range(self.num_classes):
            # print(i)
            # assigns the class_data variable to all the data that belongs to each class
            class_data = data[y == i]
            # print(class_data)
            # counts how many occurences there are for a word in emails of that class
            feature_counts = np.sum(class_data, axis=0)
            # print(feature_counts)
            # finds total count of  all words across all emails of that class
            total_counts = np.sum(feature_counts)
            # computes self.likelihoods
            self.likelihoods[i] = (feature_counts + 1) / (total_counts + num_features)


    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - For the test samples, we want to compute the log of the posterior by evaluating
        the the log of the right-hand side of Bayes Rule without the denominator (see notebook for
        equation). This can be done without loops.
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (hint: this can also be done without loops).

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        '''

        log_prior = np.log(self.priors)
        log_likelihood = np.log(self.likelihoods)

        log_posteriors = log_prior + np.dot(data, log_likelihood.T)
        classes = np.argmax(log_posteriors, axis=1)
        
        return classes
        pass

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        acc = np.mean(y == y_pred)
        return acc
        pass

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.   

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''

        confusion_mat= np.zeros((self.num_classes, self.num_classes), dtype = int)
        # print(confusion_mat.shape)
    
        # print(len(y))
        # print(len(y_pred))
        for i in range(len(y)):
            true_label = y[i]
            pred_label = y_pred[i]
            confusion_mat[true_label][pred_label] +=1
            
            

        


        return confusion_mat

        pass
