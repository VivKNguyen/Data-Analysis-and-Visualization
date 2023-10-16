'''data.py
Reads CSV files, stores data, access/filter data by variable name
Vivian Nguyen
CS 251 Data Analysis and Visualization
Spring 2023
'''

import numpy as np
import csv

class Data:
    def __init__(self, filepath=None, headers=None, data=None, header2col=None):
        '''Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each
            column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables
            (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in
                  as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0

        TODO:
        - Declare/initialize the following instance variables:
            - filepath
            - headers
            - data
            - header2col
            - Any others you find helpful in your implementation
        - If `filepath` isn't None, call the `read` method.
        '''
        # print("here!")
        self.filepath = filepath
        self.headers = headers
        self.data = data
        self.header2col = header2col
        
        if self.filepath != None:
            self.read(self.filepath)
        pass

    def read(self, filepath):
        '''Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called
        `self.data` at the end (think of this as 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if
            there should be nothing returned

        TODO:
        - Read in the .csv file `filepath` to set `self.data`. Parse the file to only store
        numeric columns of data in a 2D tabular format (ignore non-numeric ones). Make sure
        everything that you add is a float.
        - Represent `self.data` (after parsing your CSV file) as an numpy ndarray. To do this:
            - At the top of this file write: import numpy as np
            - Add this code before this method ends: self.data = np.array(self.data)
        - Be sure to fill in the fields: `self.headers`, `self.data`, `self.header2col`.

        NOTE: You may wish to leverage Python's built-in csv module. Check out the documentation here:
        https://docs.python.org/3/library/csv.html

        NOTE: In any CS251 project, you are welcome to create as many helper methods as you'd like.
        The crucial thing is to make sure that the provided method signatures work as advertised.

        NOTE: You should only use the basic Python library to do your parsing.
        (i.e. no Numpy or imports other than csv).
        Points will be taken off otherwise.

        TIPS:
        - If you're unsure of the data format, open up one of the provided CSV files in a text editor
        or check the project website for some guidelines.
        - Check out the test scripts for the desired outputs.
        '''
        self.filepath = filepath

        # open the file and reads it
        with open(self.filepath , 'r') as csvfile:
            # uses the csv reader
            csvreader = csv.reader(csvfile)
            # adds the first line of the csv file to the headers list

            # makes a list of all the headers
            labels = next(csvreader)
            self.headers= []
            # makes a list of the types 
            types = next(csvreader)

            # creating an error exception for csv files that do not have a row saying the data types
            for i in range (len(types)):
                if isinstance(types[i], str) == False:
                    raise Exception("CSV file does not have a types row")

            # adds only the numeric types to the headers list
            for i in range(len(types)):
                if types[i].strip() == "numeric":
                    self.headers.append(labels[i].strip())
            
            # makes header2col into a blank dictionary
            self.header2col = {}
            # maps each header to which column it belongs in
            for i in range (len(self.headers)):
                self.header2col[self.headers[i]] = i
			 
            self.data =[]
            
            # filters through the data so that only numeric values are added
            for row in csvreader:
                list = []
                for i in range (len(types)):
                    if types[i].strip() == "numeric":
                        list.append(row[i])
                self.data.append(list)
            
            # turns the data list into a numpy array, turns all values into floats
            self.data = np.array(self.data, dtype =float)
            
       
        pass

    def __str__(self):
        '''toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's
        called to determine what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.
        '''
        # create an empty string
        string = ""
        string += "---------------------------------------"
        # add the filename and the shape of the numpy array
        string += ("\n" + self.filepath + "  " + str(self.data.shape))
       
    #    adding the headers to the strinf
        string += ("\n"+"Headers: " + "\n")
        # making the headers into strings
        for i in range (len(self.headers)):
             string += (self.headers[i] + "    ")
    #    adding up to 3 rows of the dats
        string += "\n"
        string += "---------------------------------------"
        string += "\n"
        string += "Showing first 5 rows:"
        string += "\n"
        string += str(self.data[:5])
        

        
        return string
        


        pass

    def get_headers(self):
        '''Get method for headers

        Returns:
        -----------
        Python list of str.
        '''
        return self.headers
        pass

    def get_mappings(self):
        '''Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        '''
        return self.header2col
        pass

    def get_num_dims(self):
        '''Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        '''
        return self.data.shape[1]
        pass

    def get_num_samples(self):
        '''Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        '''
        return self.data.shape[0]
        pass

    def get_sample(self, rowInd):
        '''Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        '''
        return self.data[[rowInd]]
        pass

    def get_header_indices(self, headers):
        '''Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers`
            list.
        '''
        # create an empty list that will hold the indices
        indices = []
        for r in headers:
            for i in range(len(self.headers)):
                # if the headers parameters matches with a header, add the index to indices
                if r == self.headers[i]:
                    indices.append(i)
        return indices
            
        pass

    def get_all_data(self):
        '''Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself.
            This can be accomplished with numpy's copy function.
        '''

        copy = np.copy(self.data)
        return copy
        pass

    def head(self):
        '''Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        '''

        head =  self.data[:5]
        return head
        pass

    def tail(self):
        '''Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        '''
        tail = self.data[-5:,:]
        return tail
        pass

    def limit_samples(self, start_row, end_row):
        '''Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        '''
        self.data = self.data[start_row:end_row,:]
        

        pass

    def select_data(self, headers, rows=[]):
        '''Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified
        by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return
        column #2 of self.data. If rows is not [] (say =[0, 2, 5]), then we do the same thing,
        but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select.
                Empty list [] means take all rows

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        '''

        indices = self.get_header_indices(headers)
        
       
        if len(rows) == 0:
                
            mat = self.data[:,indices]
        else:
                
            sub_indices = np.ix_(rows,indices)
                    # print(sub_indices)
            mat= self.data[sub_indices]
        # print(self.data)
        return mat
        pass


  
