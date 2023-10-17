'''rbf_net.py
Radial Basis Function Neural Network
Vivian Nguyen
CS 251: Data Analysis Visualization
Spring 2023
'''
import numpy as np
import kmeans
import scipy.linalg

class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''
        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None

        self.k = num_hidden_units
        self.num_classes = num_classes

    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes

    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        return self.k
        

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        return self.num_classes
       

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''

        num_clusters = len(centroids)
        cluster_dists = np.zeros(num_clusters)
        for c in range(num_clusters):
            
            cluster_mask = cluster_assignments == c
            # print(cluster_mask)
            cluster_data = data[cluster_mask]

            
            dists = kmeans_obj.dist_pt_to_centroids(cluster_data, centroids[c].reshape(1, -1))

            
            cluster_dists[c] = np.mean(dists)

        return cluster_dists
        

    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        # self.prototypes = kmeans_obj.cluster_batch(k = self.k, n_iter = 5)
        # self.sigmas = self.avg_cluster_dist(data, kmeans_obj.get_centroids(), self.prototypes, kmeans_obj)

        # Determine prototypes using cluster_batch
        num_prototypes = self.k
        k_means = kmeans.KMeans(data=data)
        k_means.cluster_batch(k= num_prototypes, n_iter=5)
        self.prototypes = k_means.get_centroids()
        
        # self.sigmas = average distance between each cluster center and data points that are assigned to it.
        self.sigmas = self.avg_cluster_dist(data=data, centroids=self.prototypes, cluster_assignments=k_means.get_data_centroid_labels(), kmeans_obj=k_means)
      

    def linear_regression(self, A, y):
        '''Performs linear regression
        CS251: Adapt your SciPy lstsq code from the linear regression project.
        CS252: Adapt your QR-based linear regression solver

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: Remember to handle the intercept ("homogenous coordinate")
        '''
        Ahat = np.hstack((A, np.ones((A.shape[0],1))))
        # print(np.mean(np.isnan(Ahat) == True))
        # print(np.mean(np.isnan(A) == True))
        # print(np.mean(np.isnan(y) == True))
        c,_,_,_ = scipy.linalg.lstsq(Ahat,y)
        
        return c
       

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''
        # initializes the prototypes and sigmas
        # self.initialize(data)
        # print(self.prototypes.shape)
        # print(self.prototypes)
        # print(data)
        eps = 1e-8
        activations = np.zeros((data.shape[0], self.prototypes.shape[0]))
        for i in range(data.shape[0]):
            for j in range(self.prototypes.shape[0]):
                dist = np.linalg.norm(data[i] - self.prototypes[j])
                activations[i, j] = np.exp(-(dist**2)/(2*self.sigmas[j]**2 + eps))
        return activations

        
       

    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''
        h1 = np.hstack((hidden_acts, np.ones((hidden_acts.shape[0],1))))
        # print(h1)
        # print(self.wts.shape)
        # print(self.wts)
        # print(h1)
        return h1 @self.wts
        

        
        

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        # initializing prototypes and sigmas
        self.initialize(data)
        # print(y)
        y = y.astype(int)
        # print(y)
        # convert y to one-hot, from lecture notes 
        y_one_hot = np.zeros((len(y),self.num_classes))
        for i in range(len(y)):
            y_one_hot[i, y[i]] = 1

        H = self.hidden_act(data)
        # print(np.mean(np.isnan(H) == True))
        self.wts = self.linear_regression(H, y_one_hot)

        # print(self.wts.shape)


        

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''

        H = self.hidden_act(data)
        Z = self.output_act(H)
        # print(Z)
        # for each sample
        y_pred = np.zeros((Z.shape[0]))
        for i in range(Z.shape[0]):
            max_idx = np.argmax(Z[i])
            y_pred[i] = max_idx
        # print(y_pred.shape)
        return y_pred
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

        return np.mean(y == y_pred)
        pass
