'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Vivian Nguyen
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


class Transformation(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''

        analysis.Analysis.__init__(self, data=data)
        self.orig_dataset = orig_dataset
        pass

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''
        # creating a new Data object based on the headers in the parameter
        new_data = self.orig_dataset.select_data(headers)
        # create new header2col
        new_header2col = {}
        ind = 0
        for i in headers:
            new_header2col[i] = ind
            ind +=1
        # fill in valid values for the Data constructor except filepath
        self.data = data.Data(headers = headers, data = new_data, header2col= new_header2col)
        pass

    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        # create a column of ones that has as many rows as the array
        ones = np.ones((self.data.get_num_samples(),1))
        x =self.data.get_all_data()
        # print(ones)
        # add it to the data
        arr = np.hstack((x, ones))
        return arr
        pass

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        magnitudes.append(1)
        # print(magnitudes)
        mags = np.array(magnitudes).reshape(1,len(magnitudes)).T

        translate = np.eye(self.data.get_num_dims()+1, self.data.get_num_dims())


        trans = np.hstack((translate,mags))
        
        return trans

        pass

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''

        features = self.data.get_num_dims() + 1
        scale = np.eye(self.data.get_num_dims() + 1)
        # print(scale)
        magnitudes.append(1)

        scale[np.arange(self.data.get_num_dims() + 1), np.arange(self.data.get_num_dims() + 1)] = magnitudes
       
        return scale
        pass

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''

        dh = self.get_data_homogeneous().T
        trans = self.translation_matrix(magnitudes)
        # print(trans)
        
        dobj = (trans @ dh).T
        dobj =np.delete(dobj, dobj.shape[1]-1, 1)

        self.data = data.Data(headers = self.data.get_headers(), data = dobj, header2col= self.data.get_mappings())
        return dobj


        pass

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        dh = self.get_data_homogeneous().T

        scale = self.scale_matrix(magnitudes)

        scale_obj = (scale @ dh).T

        scale_obj =np.delete(scale_obj, scale_obj.shape[1]-1, 1)
        
        self.data = data.Data(headers = self.data.get_headers(), data = scale_obj, header2col= self.data.get_mappings())
        return scale_obj


        pass

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''

        dat = self.get_data_homogeneous().T
      
        transf_obj = (C @ dat).T
        
        transf_obj =np.delete(transf_obj, transf_obj.shape[1]-1, 1)
        
        self.data = data.Data(headers = self.data.get_headers(), data = transf_obj, header2col= self.data.get_mappings())
        return transf_obj
        pass

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''

        mn = self.min(self.data.get_headers())
        min = np.min(mn)
        # print(mn)
        # print(min)
        mx = self.max(self.data.get_headers())
        max = np.max(mx)
        
        s_variable = 1/(max-min)
        tmagnitudes = []
        smagnitudes = []

        for i in range (self.data.get_num_dims()):
            tmagnitudes.append(-min)
            smagnitudes.append(s_variable)
        
        
        translate_mat = self.translation_matrix(tmagnitudes)
        scale_mat = self.scale_matrix(smagnitudes)
        
        c = scale_mat @ translate_mat
        
        return self.transform(c)
        

        pass

    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''

        mins = []
        maxs = []

        for i in  self.min(self.data.get_headers()):
            mins.append(i*-1)
        
        for i in  self.max(self.data.get_headers()):
            maxs.append(i)
        
        scale_var = []

        translate_mat = self.translation_matrix(mins)
        
        for i in range(len(maxs)):
            scale_var.append(1/(maxs[i]+mins[i]))

        scale_mat = self.scale_matrix(scale_var)
        
        c = scale_mat @ translate_mat

        return self.transform(c)
       
        pass

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''

        rot_mat = np.eye(4)

        rads = np.deg2rad(degrees)
        if header == self.data.get_headers()[0]:
            rot_mat[1,1] = np.cos(rads)
            rot_mat[1,2] = -np.sin(rads)
            rot_mat[2,1] = np.sin(rads)
            rot_mat[2,2] = np.cos(rads)
        elif header == self.data.get_headers()[1]:
            rot_mat[0,0] = np.cos(rads)
            rot_mat[0,2] = np.sin(rads)
            rot_mat[2,0] = -np.sin(rads)
            rot_mat[2,2] = np.cos(rads)
        else:
            rot_mat[0,0] = np.cos(rads)
            rot_mat[0,1] = -np.sin(rads)
            rot_mat[1,0] = np.sin(rads)
            rot_mat[1,1] = np.cos(rads)

        return rot_mat


        pass

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''

        dh = self.get_data_homogeneous().T
        rot_mat = self.rotation_matrix_3d(header, degrees)

        rotated = (rot_mat @ dh).T
        rotated =np.delete(rotated, rotated.shape[1]-1, 1)
        
        self.data = data.Data(headers = self.data.get_headers(), data = rotated, header2col= self.data.get_mappings())

        return rotated
        pass

    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        
        
        headers = self.data.get_headers()
        colors = []
        headerslist =[]
        for i in headers:
            if i == ind_var:
                headerslist.append(i)
            if i == dep_var:
                headerslist.append(i)
            if i == c_var:
                colors.append(i)
        arr = self.data.select_data(headerslist)
        col = self.data.select_data(colors)
        x = arr[:,0]
        y = arr[:,1]

       
        color_map = palettable.colorbrewer.sequential.Greys_9
        plt.scatter(x,y, c = col, cmap=color_map.mpl_colormap, edgecolor='gray')
        cbar = plt.colorbar()
        cbar.set_label(c_var)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.title(title)
        plt.show()
        pass


    # EXTENSION 1
    def scatter_4d (self, x_var, y_var, z_var, c_var, title=None):
        '''Creates a 4D scatter plot with a color scale representing the 4th dimension.

        Parameters:
        -----------
        x_var: str. Header of the variable that will be plotted along the X axis.
        y_var: Header of the variable that will be plotted along the Y axis.
        z_var: Header of the variable that will be plotted along the Z axis
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''

        headers = self.data.get_headers()
        
        colors = []
        headerslist =[]
        for i in headers:
            if i == x_var:
                headerslist.append(i)
            if i == y_var:
                headerslist.append(i)
            if i ==z_var:
                headerslist.append(i)
            if i == c_var:
                colors.append(i)
        
        arr = self.data.select_data(headerslist)
        col = self.data.select_data(colors)
        x = arr[:,0]
        y = arr[:,1]
        z = arr[:,2]

        color_map = palettable.colorbrewer.sequential.Greys_9
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(111, projection='3d')


        cmap=color_map.mpl_colormap
        norm = mpl.colors.Normalize()

        ax.scatter(x, y, z, c = col, cmap= cmap, edgecolor='gray')
        ax.set_xlabel(headerslist[0], labelpad=7)
        ax.set_ylabel(headerslist[1], labelpad=7)
        ax.set_zlabel(headerslist[2], labelpad=7)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad =0.1)
        cbar.set_label(c_var)

        plt.title(title)
        plt.show()

    # EXTENSION 2
    def scatter_5d (self, x_var, y_var, z_var, c_var, size_var, title=None):
        '''Creates a 4D scatter plot with a color scale representing the 4th dimension.

        Parameters:
        -----------
        x_var: str. Header of the variable that will be plotted along the X axis.
        y_var: Header of the variable that will be plotted along the Y axis.
        z_var: Header of the variable that will be plotted along the Z axis
        c_var: Header of the variable that will be plotted along the color axis.
        size_var: Header of the variable that will be plotted with the size of the markers.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''

        headers = self.data.get_headers()
        
        colors = []
        headerslist =[]
        for i in headers:
            if i == x_var:
                headerslist.append(i)
            if i == y_var:
                headerslist.append(i)
            if i ==z_var:
                headerslist.append(i)
            if i == c_var:
                colors.append(i)
            if i == size_var:
                headerslist.append(i)
        
        arr = self.data.select_data(headerslist)
        col = self.data.select_data(colors)
        x = arr[:,0]
        y = arr[:,1]
        z = arr[:,2]
        sizes = 50 * (arr[:, 3] - arr[:, 3].min()) / (arr[:, 3].max() - arr[:, 3].min())

        color_map = palettable.colorbrewer.sequential.Greys_9
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(111, projection='3d')


        cmap=color_map.mpl_colormap
        norm = mpl.colors.Normalize()

        font1 = {'family':'serif','color':'black','size':15}

        ax.scatter(x, y, z, c = col,s =sizes, cmap= cmap, edgecolor='gray')
        ax.set_xlabel(headerslist[0], labelpad=7)
        ax.set_ylabel(headerslist[1], labelpad=7)
        ax.set_zlabel(headerslist[2], labelpad=7)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad =0.1)
        cbar.set_label(c_var)

        plt.title(title, fontdict = font1)
        plt.show()

    # EXTENSION 3

    def zscore_normalize(self, headers, rows = []):
        '''Computes the zscore for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Zscore values for each of the selected header variables

        '''
        mean = self.mean(headers,rows)
        std = self.std(headers, rows)
        x = np.subtract(self.data.select_data(headers,rows),mean)
        z = np.divide(x,std)

        self.data = data.Data(headers = self.data.get_headers(), data = z, header2col= self.data.get_mappings())

        return z
    
    # EXTENSION 4
    def rotationmat_2d(self, degrees):
        rot_mat = np.eye(3)

        rads = np.deg2rad(degrees)
        
        rot_mat[0,0] = np.cos(rads)
        rot_mat[0,1] = -np.sin(rads)
        rot_mat[1,0] = np.sin(rads)
        rot_mat[1,1] = np.cos(rads)
        
            

        return rot_mat
    
    def rotate_2d(self, degrees ):
        '''Rotates the projected data by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!'''
        
        dh = self.get_data_homogeneous().T
        
        rot_mat = self.rotationmat_2d(degrees)
        

        rotated = (rot_mat @ dh).T
        rotated =np.delete(rotated, rotated.shape[1]-1, 1)
        
        self.data = data.Data(headers = self.data.get_headers(), data = rotated, header2col= self.data.get_mappings())

        return rotated