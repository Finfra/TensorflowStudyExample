
import numpy as np
def xtab(*cols):
    '''
    # References : https://gist.github.com/alexland/d6d64d3f634895b9dc8e
    returns:
        (i) xt, NumPy array storing the xtab results, number of dimensions is equal to 
                the len(args) passed in
        (ii) unique_vals_all_cols, a tuple of 1D NumPy array for each dimension 
                in xt (for a 2D xtab, the tuple comprises the row and column headers)
        pass in:
            (i) 1 or more 1D NumPy arrays of integers
            (ii) if wts is True, then the last array in cols is an array of weights
            
    if return_inverse=True, then np.unique also returns an integer index 
    (from 0, & of same len as array passed in) such that, uniq_vals[idx] gives the original array passed in

    higher dimensional cross tabulations are supported (eg, 2D & 3D)

    cross tabulation on two variables (columns):

    >>> q1 = np.array([7, 8, 8, 8, 5, 6, 4, 6, 6, 8, 4, 6, 6, 6, 6, 8, 8, 5, 8, 6])
    >>> q2 = np.array([6, 4, 6, 4, 8, 8, 4, 8, 7, 4, 4, 8, 8, 7, 5, 4, 8, 4, 4, 4])

    >>> uv, xt = xtab(q1, q2)
    >>> uv
        (array([4, 5, 6, 7, 8]), array([4, 5, 6, 7, 8]))

    >>> xt
        array([[2, 0, 0, 0, 0],
                     [1, 0, 0, 0, 1],
                     [1, 1, 0, 2, 4],
                     [0, 0, 1, 0, 0],
                     [5, 0, 1, 0, 1]], dtype=uint64)
        '''
    apply_wt=False
    if not all(len(col) == len(cols[0]) for col in cols[1:]):
        raise ValueError("all arguments must be same size")
    if len(cols) == 0:
        raise TypeError("xtab() requires at least one argument")
    fnx1 = lambda q: len(q.squeeze().shape)
    if not all([fnx1(col) == 1 for col in cols]):
        raise ValueError("all input arrays must be 1D")
    if apply_wt:
        cols, wt = cols[:-1], cols[-1]
    else:
        wt = 1
    uniq_vals_all_cols, idx = zip( *(np.unique(col, return_inverse=True) for col in cols) )
    shape_xt = [uniq_vals_col.size for uniq_vals_col in uniq_vals_all_cols]
    dtype_xt = 'float' if apply_wt else 'uint'
    xt = np.zeros(shape_xt, dtype=dtype_xt)
    np.add.at(xt, idx, wt)
    return uniq_vals_all_cols, xt

if __name__ == "__main__": 
    q1 = np.array([2,3,3,4])
    q2 = np.array([2,2,3,4])
    uv, xt = xtab(q1, q2)
    print(q1)
    print(q2)
    print(uv)    
    print(xt)