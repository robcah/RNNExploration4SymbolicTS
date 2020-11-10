import numpy as np


def SymbolicTS(string='AB', datalen=2200):
    '''Creation of a Symbolic time-series based on a string-seed, which will be 
    multiplied until achieve a length above datalen parameter
    
    Parameters
    ----------
    string : str
        String "seed" which is multiplied till reach a length datalen.
    datalen : int
        Integer stating the aproximate length of the symbolic time-series. 
        
    Return
    ------
    str :
        The symbolic time-series resulting of looping the string "seed" to an approximate datalen length.
    
    Examples
    --------
    >>> SymbolicTS('ABC', 20)
    'ABCABCABCABCABCABCABC'
    
    >>> SymbolicTS('abcdefg', 10)
    'abcdefgabcdefg'
        
    '''
    datalen = int(datalen)
    factor = int(np.ceil(datalen / len(string)))
    seq =  string * factor
    
    return np.array(list(seq))


def SymbolList(sym_ts=None):
    '''Create the list of unique symbols within a symbolic time-series.
    
    Parameters
    ----------
    sym_ts : str
        Symbolic time series, if None the value will be assigned by the default settings of SymbolicTS(). 
        
    Return
    ------
    list :
        Sorted list in alphabetical order of the symbols used in the given symbolic time-series
    
    Examples
    --------
    >>> SymbolList('ACBAAABBC')
    ['A', 'B', 'C']
        
    '''
    if sym_ts is None: sym_ts = SymbolicTS()
    
    return sorted(list(set(sym_ts)))


def DataOneHotEncode(sym_ts=None, forecasting=100, X_sequence=100):
    '''Produces a one hot encoding from the symbolic time-series, leaving a sequence 
    of length given by forecasting as a prediction test sequence.
    
    Parameters
    ----------
    sym_ts : str
        Symbolic time series, if None the value will be assigned by the default settings of SymbolicTS().
    forecasting : int
        Length of the symbolic time-series left to be predicted, default 100.
    X_sequence :  int
        Length of the window for data "X" sequence, namely the length of ordered inputs to produce a "y" output.
        
    Return
    ------
    array : dataX
        One hot encode array with shape (m, n, p) where m is the symbolic time-series "sym_ts", length minus 
        the parameter "forecasting" sequence, n stand for the X_sequence; and p stands for the number of 
        symbols used in the time-series.
    array : datay
        One hot code array with shape (m, p) giving the inmediate output value after each dataX input sequence.
    array : dataTest
        String with length q defined by parameter "forecasting", slice of the last q values of the symbolic 
        time-series "sym_ts".
    
    Examples
    --------
    >>> DataOneHotEncode('ABABABAB', 2, 2)
    ([array([[1, 0], [0, 1]]), array([[0, 1], [1, 0]]), array([[1, 0], [0, 1]]), array([[0, 1], [1, 0]])],
    [array([1, 0]), array([0, 1]), array([1, 0]), array([0, 1])],
    'AB')   
    '''
    
    if sym_ts is None: sym_ts = SymbolicTS()
    dataXy = sym_ts[:-forecasting]
    dataTest = sym_ts[-forecasting:]
    symbols = SymbolList(dataXy)
    onehot = np.array([[0 if symbol != char else 1 for symbol in symbols] for char in dataXy])
    
    dataX = []
    datay = []

    for i in range(0, len(onehot)-X_sequence, 1):
        s_in = onehot[i:i+X_sequence]
        s_out = onehot[i+X_sequence]
        dataX.append(s_in)
        datay.append(s_out)
    
    return dataX, datay, dataTest