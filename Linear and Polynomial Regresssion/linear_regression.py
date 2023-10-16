'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Vivian Nguyen
CS251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean SEE. float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        self.c = None
        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression by using Scipy to solve the least squares problem y = Ac
        for the vector c of regression fit coefficients. Don't forget to add the coefficient column
        for the intercept!
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''

        vars = []
        for i in ind_vars:
            vars.append(i)
        vars.append(dep_var)
        
        data = self.data.select_data(vars)
        
        
        a = data[:,:len(vars)-1]
        
        y = data[:,-1]
        y = y.reshape((y.shape[0],1))
        ahat = np.hstack((np.ones((a.shape[0],1)), a))
        c, _, _, _ = scipy.linalg.lstsq(ahat,y)
        
        linreg = scipy.linalg.lstsq(ahat,y)
        yhat = ahat @ c # predicted values
        rs = y - yhat
        R2 = 1 - np.sum(rs**2)/np.sum((y-y.mean())**2)
        self.slope = c[1:,]
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.A = a
        self.intercept = float(c[:1,])
        self.y = y
        self.R2 = R2
        self.c = linreg[0]
        y_pred = self.predict()
        
        
        self.residuals = self.compute_residuals(y_pred)
        self.intercept = float(c[:1,])
        
        
        self.mse = self.compute_mse()

        
        

        pass

    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        
        list = [self.intercept]
        for i in self.slope:
            list.append(float(i))
        
        mat = np.array(list, ndmin=2).T

        if self.p ==1:
            if X is not None:
                X = np.hstack((np.ones((X.shape[0],1)), X))
                y_pred = X @ mat
            else:
                ahat = np.hstack((np.ones((self.A.shape[0],1)), self.A))
                y_pred = ahat @ mat
        else:
            if X is not None:
                X = np.hstack((np.ones((X.shape[0],1)), X))
                y_pred = X @ mat
            else:
                poly_mat = self.make_polynomial_matrix(self.A,self.p)
                poly_mathat = np.hstack((np.ones((poly_mat.shape[0],1)), poly_mat))
                y_pred = poly_mathat @ mat
                
        
        return y_pred

        pass

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''

        return self.R2
        pass

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''

        residuals = self.y - y_pred
        return residuals
        pass

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''
        mse = np.mean(self.residuals **2)
        return mse
        pass

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''

        super().scatter(ind_var, dep_var, title)
        xmin = self.A.min()
        xmax = self.A.max()
        line_x = np.linspace(xmin, xmax, 100)
        line_y = (line_x * self.slope) + self.intercept
        
        if self.p > 1:
            
            lineM_x = self.make_polynomial_matrix(line_x.reshape((line_x.shape[0],1)), self.p)
            
            yvals = lineM_x @ self.slope + self.intercept   
            
            print('linemx ',lineM_x.shape)
            print('slope ', self.slope.shape)
            print('yvals ',yvals.shape)
            
            plt.plot(line_x,yvals )
            
        
            
        else:
            
            plt.plot(line_x, line_y.T, label = 'linear regression')
            
        plt.title(title + ' $R^2$ = ' + str("{:.2f}".format(self.R2)))
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        print(self.mse)
        pass

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        fig, axes = super().pair_plot(data_vars, fig_sz)

        for i in range(len(data_vars)):
            for j in range(len(data_vars)):
                # so that the main diagonal can be used for histogram
                if i != j:
                    self.linear_regression([data_vars[i]], data_vars[j])
                    line_x = np.linspace(np.min(axes[i, j].get_xlim()), np.max(axes[i, j].get_xlim()), 100)
                    line_y = (line_x * self.slope) + self.intercept
                    axes[i,j].plot(line_x,line_y.T)
                    axes[i,j].set_title(f'{data_vars[j]} & {data_vars[i]} R^2 = {self.R2:.2f}', fontsize = 8)

        

        if hists_on_diag:
            vars_num = len(data_vars)
            for i in range(vars_num):
                axes[i, i].remove()
                axes[i, i] = fig.add_subplot(vars_num, vars_num, i*vars_num+i+1)
                if i < vars_num-1:
                    axes[i, i].set_xticks([])
                else:
                    axes[i, i].set_xlabel(data_vars[i])
                if i > 0:
                    axes[i, i].set_yticks([])
                else:
                    axes[i, i].set_ylabel(data_vars[i])

                axes[i, i].hist(self.data.select_data([data_vars[i]]), bins=10)
                axes[i, i].set_title(f'{data_vars[i]} histogram, 10 bins', fontsize=10)


        pass

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        should take care of that.
        '''

        mat = np.array(A, dtype= float)
        
        for i in range(2,p+1):

            mat = np.hstack((mat, A**i))
        
        return mat
        pass

    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''

        self.p = p
        
        vars = [ind_var, dep_var]
        data = self.data.select_data(vars)
        a = data[:,0].reshape((data.shape[0],1))
        
        poly_matrix = self.make_polynomial_matrix(a, self.p)
        
        
        y = data[:,1]
        y = y.reshape((y.shape[0],1))
        ahat = np.hstack((np.ones((poly_matrix.shape[0],1)), poly_matrix))

        c, _, _, _ = scipy.linalg.lstsq(ahat,y)
        
        yhat = ahat @ c # predicted values
        rs = y - yhat
        R2 = 1 - np.sum(rs**2)/np.sum((y-y.mean())**2)
        self.slope = c[1:,]
        self.ind_vars = ind_var
        self.dep_var = dep_var
        self.A = a
        self.intercept = float(c[:1,])
        self.y = y
        self.R2 = R2
        linreg = scipy.linalg.lstsq(ahat,y)
        y_pred = self.predict()
        
        self.c = c
        self.residuals = self.compute_residuals(y_pred)
        self.intercept = float(c[:1,])
        
        
        self.mse = self.compute_mse()
        
        pass

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept
        pass

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor. 
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.slope = slope
        self.intercept = intercept
        self.p = p
        pass

    

