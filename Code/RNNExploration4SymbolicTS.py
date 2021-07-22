import sys
import numpy as np
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, RNN, Flatten
from tensorflow.keras.layers import SimpleRNN as sRNN
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import optimizers
import pandas as pd
from pathlib import Path
from os.path import join, exists
from os import listdir, getcwd, remove
import textdistance as td
import time as T

from RNNNovelArchitectures import *

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


def DataOneHotEncode(sym_ts=None, validation=10, X_window=10):
    '''Produces a one hot encoding from the symbolic time-series, leaving a sequence 
    of length given by forecasting as a prediction test sequence.
    
    Parameters
    ----------
    sym_ts : str
        Symbolic time series, if None the value will be assigned by the default settings of SymbolicTS().
    validation : int
        Length of the symbolic time-series left to be predicted, default 100.
    X_window :  int
        Length of the window for data "X" sequence, namely the length of ordered inputs to produce a "y" output.
        
    Return
    ------
    array : dataX
        One hot encode array with shape (m, n, p) where m is the symbolic time-series "sym_ts", length minus 
        the parameter "validation" sequence, n stand for the X_window; and p stands for the number of 
        symbols used in the time-series.
    array : datay
        One hot code array with shape (m, p) giving the inmediate output value after each dataX input sequence.
    array : dataTest
        String with length q defined by parameter "validation", slice of the last q values of the symbolic 
        time-series "sym_ts".
    
    Examples
    --------
    >>> DataOneHotEncode('ABABABAB', 2, 2)
    ([array([[1, 0], [0, 1]]), array([[0, 1], [1, 0]]), array([[1, 0], [0, 1]]), array([[0, 1], [1, 0]])],
    [array([1, 0]), array([0, 1]), array([1, 0]), array([0, 1])],
    'AB')   
    '''
    
    if sym_ts is None: sym_ts = SymbolicTS()
    dataXy = sym_ts[:-validation]
    dataTest = sym_ts[-validation:]
    symbols = SymbolList(dataXy)
    onehot = np.array([[0 if symbol != char else 1 for symbol in symbols] for char in dataXy])
    
    dataX = []
    datay = []

    for i in range(0, len(onehot)-X_window, 1):
        s_in = onehot[i:i+X_window]
        s_out = onehot[i+X_window]
        dataX.append(s_in)
        datay.append(s_out)
    
    return dataX, datay, dataTest
    

class EarlyStoppingValue(Callback):
    '''Child class based on tensorflow.keras.callbacks.Callback, with the finality to produce an early 
    stop depending on the value of a monitored parameter such as "acc", "loss", "val_acc", "val_loss", 
    among others.
    
    Attribute
    ----------
    monitor : str
        Monitored parameter which value will trigger the early stop.
    stop_value : float
        Value that once exceeded or becoming inferior, depending whether the criteria, will trigger the early stop.
    verbose :  int
        Quantity of information displayed, values from 0 to 2, 0 will not display any information, 2, will display 
        all information available.
        
    Return
    ------
    boolean : self.model.stop_training
        This is the inherited argument from class "Callback" that will stop the training once the value reached.
    '''
    
    def __init__(self, monitor='acc', stop_value=1.0, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.stop_value = stop_value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        # retrieve the value of the monitored parameter and check whether its output is activated.
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires {} available!".format(self.monitor), RuntimeWarning)
        
        # Analyse if the minitored paramater is either accuracy or loss to determine the relation to the stopping value.
        # If the monitored parameter is not recognised it request to define the relation to the stop_value.
        if 'acc' in self.monitor:
            conditional = current > self.stop_value
            operator = '>'
        elif 'loss' in self.monitor:
            conditional = current < self.stop_value
            operator = '<'
        else:
            operator = input('Specify relation of {} to {} "<" or ">":'.format(self.monitor, self.stop_value))
            exec('conditional = current {} self.value'.format(operator))
        
        if conditional:
            if self.verbose > 0:
                print("Epoch {}: early stopping at {} {} {}".format(epoch, self.monitor, operator, self.stop_value))
            self.model.stop_training = True
            
class TimeHistory(Callback):
    '''Child class based on tensorflow.keras.callbacks.Callback, with the finality to measure the process time taken by epoch.
        
    Return
    ------
    list : self.times
        Value in seconds of the processing time of the finalised epoch.
    '''
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = T.time() # Initial time in seconds

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(T.time() - self.epoch_time_start) # time delta when finished
            
class simpleRNN(object):
    '''Class to build, train and forecast with Keras inherent LSTM architecture.
    
    Attributes
    ----------
    X_shape : array 
        Array of dimensions (3,) providing the shape of the input training sequences.
    y_shape : array
        Array of dimensions (1,) providing the output training values.
    n_layers : int
        Number of hidden layers to be contain in the Neural Network. Defult value 1.
    units_per_layer : int
        Fixed number of units (also called cells) to be contained in each layer. Defult value 100.
    dropout :  float
        Fraction of the input and recurrent conections to dropout. Default value 0.2.
    sym_ts : str
        String sequence representing a symbolic time-series. Default value "abab..." 
        up to a length of 1000 characters.
    lr : float
        Learning rate of the optimization function. Default value 0.001 to be used 
        in Adam optimisation.
    process_time : float
        Time in seconds in which the epoch was processed.
    
    
    Methods
    -------
    Build()
        It builds and compiles the model.
    Checkpoint()
        It creates the checkpoint callback and saves the weights of succesful epochs 
        (when the loss is reduced). This method also can activate the early stop either 
        by patience using tensorflow.keras.callbacks.EarlyStopping or by EarlyStoppingValue,
        which waits until a monitored optimisation parameter reaches a value to stop. 
        This method also records the processing time of the epoch to be added as the attribute:
        process_time.
    Train():
        Executes the fit method of the compiled model.
        
    '''
    
    def __init__(self, X_shape=None, y_shape=None, n_layers=1, units_per_layer=100, 
                 dropout=0.2, sym_ts='ab'*500, lr=0.001, architecture='LSTM'):
        '''
        Parameters
        ----------
        X_shape : array 
            Array of dimensions (3,) providing the shape of the input training sequences.
        y_shape : array
            Array of dimensions (1,) providing the output training values.
        n_layers : int
            Number of hidden layers to be contain in the Neural Network. Defult value 1.
        units_per_layer : int
            Fixed Number of cells (also called units) to be contained in each layer. Defult value 100.
        dropout :  float
            Fraction of the input and recurrent conections to dropout. Default value 0.2.
        sym_ts : str
            String sequence representing a symbolic time-series. Default value "abab..." 
            up to a length of 1000 characters.
        lr : float
            Learning rate of the optimization function. Default value 0.001 to be used 
            in Adam optimisation.
        '''
        self.sym_ts = sym_ts
        self.n_layers =  n_layers
        self.units_per_layer = units_per_layer
        self.dropout = dropout
        self.X_shape = X_shape
        self.y_shape = y_shape
        self.architecture = architecture
        self.lr = lr
            
    def Build(self):
        '''Calls for the model building with the settings stated in the class, 
        compiles it with Adam optimiser and categorical crossentropy as loss function.        
        '''
        codestr = f'{self.architecture}U{self.units_per_layer}L{self.n_layers}'
        self.wfolder = T.strftime(f'{codestr}_%Y%m%d_%H%M%S') # subfolder for weights
        
        adam = optimizers.Adam(learning_rate=self.lr, clipvalue=1.0)
        clear_session()
        self.model = Model(self.X_shape, self.y_shape, self.n_layers, self.units_per_layer, 
                           self.dropout, self.architecture)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
            
    def Checkpoint(self, earlystop=True, param='acc', stopval=0.99, patience=0):
        '''Defines the callback settings and early stop criteria.
        
        Parameters
        ----------
        earlystop : bool
            Activates the feature of early stop and the measure of processing 
            time of each epoch, defalut True.
        param : str
            Parameter to be monitored for either early stop or patience, 
            default accuracy (acc)
        stopval : float
            Value that once reached will trigger the early stop, default 0.99.
        patience : int
            Number of epochs to trigger early stop if no significant change 
            on the monitored paramater is reported.
            
        Returns
        -------
        list
            The list will containg the Callback ModelCheckpoint, saving the 
            weights in a HDF5 file in the subfolder "weights" if the parameter loss 
            shows improvement. Depending on the parameters the list might additionally 
            contain: (i) An early stop class, either by patience or by reaching a given 
            value in a monitored parameter; (ii) a TimeHistory callback object, which 
            measures the processing time of the epoch and saves it as an attribute in 
            of the simpleLSTM class. 
            
        '''
        cwd = getcwd()
        folder = r'weights'
        folderpath = join(cwd, folder, self.wfolder)
        filepath = join(folderpath, r'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5')
        
        def checkpoint_def():
            return ModelCheckpoint(filepath, monitor='loss', 
                verbose=1, save_best_only=True, mode='min')
        
        # Tries to find folder otherwise it creates it
        if not exists(folderpath):
            Path(folderpath).mkdir(parents=True, exist_ok=True)
        
        checkpoint = checkpoint_def()
        
        self.process_time = TimeHistory()
        
        # Options to early stop
        if earlystop:
            es = EarlyStoppingValue(monitor=param, stop_value=stopval, verbose=1)
            return [checkpoint, es, self.process_time]
        
        if patience:
            es = EarlyStopping(monitor='loss', mode='min', verbose=1, min_delta=0.01, patience=patience)
            return [checkpoint, es]
        
        return [checkpoint]
    
    def Train(self, X=None, y=None, model=None, epochs=5, batch_size=128, 
              callbacks=None, validation_split=0.05, param='loss', stopval=0.1, patience=0): 
        '''Fitting of the attribute model to the input (X) and output (y) data.
        
        Parameters
        ----------
        X : array
            Input data.
        y : array
            Output data.
        model : tensorflow.keras.Sequential()
            Recurrent Neural Network model to be fitted to data X and y.
        epochs : int
            Number of epochs (an epoch is the fitting the model throughout all the X and y data)
        batch_size : int
            Number of samples that will be propagated throught the network per iteration.
        callbacks : Callback object or list of Callback objects
            Callback settings, including saving weight file, early stopping or 
            time process recording
        validation_split : float
            Fraction of data to take for validation.
        param : str
            Parameter to monitor for early stopping.
        stopval : float
            Reference value to trigger early stopping.
        patience : int
            Number of epochs without significant change to trigger early stopping.
        '''
        
        if X is None or y is None:
            data = Data(sym_ts=self.sym_ts)
            self.X, self.y = data.Xy()
        else:
            self.X, self.y = X, y
        
        if model is None: model = self.model
        if callbacks is None: callbacks = self.Checkpoint(param=param, stopval=stopval, 
                                                          patience=patience)
        
        self.h = model.fit(self.X, self.y, validation_split=validation_split, 
                           epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        
    def Forecast(self, model=None, dataY=None, dataTest=None, 
                 filename=None, delnonoptimals=False):
        self.delnonoptimals = delnonoptimals
        '''This method generates a forecast through the population of its weights and 
        biases from the epoch with best results.
        
        Parameters
        ----------
        model : tensorflow.keras.Sequential()
            Recurrent Neural Network model used to forecast.
        
        dataY : array
            Output data
        
        dataTest : array
            Test data to measure accuracy of forecast.
            
        filename : string
            Name of the file with the fitted weights and biases that reached the early 
            stop criterium.
            
        Returns
        -------
        jw : float
            Jaro-Winkler distance between dataTest and the output forecast sequence.
        dl : float
            Damerau-Levenshtein distance between dataTest and the output forecast sequence
        seq_test : string
            Original test data.
        seq_out : string
            Forecast sequence.
        '''

        if model is None: model = self.model
        if dataY is None and dataTest is None: 
            _, dataY, dataTest = data.Data()
        seq_test = ''.join(dataTest) 
        
        if not filename:
            optimisedweights = History(self.h, self.wfolder, 
                                   delnonoptimals=self.delnonoptimals)
        else:
            optimisedweights = filename
            
        print(f'Using: {optimisedweights}')

        n_vocab = len(SymbolList(self.sym_ts)) 
        model.load_weights(optimisedweights)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        # reverse mapping that we can use to convert the integers back to characters
        int_to_char = SymbolList(self.sym_ts)

        # make prediction
        pattern = dataY[-100:]
        print("Forecast from:")
        sym = []
        for i in pattern:
            idx = np.argwhere(i)[0][0]
            sym.append(int_to_char[idx])
            
        seed = ''.join(sym)
        print('"{}"'.format(seed))
        # generate characters
        test_len = len(seq_test) # change for length of dataTest
        seq_out = ''
        shape = pattern.shape
        for i in range(test_len):
            x = np.reshape(pattern, (1, shape[0], shape[1]))
            prediction = model.predict(x, verbose=0)
            index = np.argmax(prediction)#[0][0]
            result = int_to_char[index]
            seq_out += result
            pattern = np.concatenate((pattern, prediction))
            pattern = pattern[1:len(pattern)]
            
        jw = td.jaro_winkler.normalized_similarity(seq_test, seq_out)
        dl = td.damerau_levenshtein.normalized_similarity(seq_test, seq_out)
        print(f'\nValidation:\t{seq_test}\nPredicted:\t{seq_out}\nJaro-Winkler: {jw:.4f}\tDamerau-Levenshtein: {dl:.4f}')
        
        return jw, dl, seq_test, seq_out
    
def Model(X_shape=None, y_shape=None, n_layers=1, units_layer=100, dropout=0.2, 
          architecture='LSTM'):
    '''Function to produce a Recurrent Neural Network customising some settings.
        
        Parameters
        ----------
        X_shape : array
            This array is contains the shape of the input data.
        y_shape: array
            This array contains the shape of the output data.
        n_layers : int
            Depth of the Recurrent Neural Network.
        units_layer : int
            number of cells or hidden states produced within the layer of the 
            Recurrent Neural Network
        dropout : float
            Fraction of the conectios to dropout during trainning
        architecture :  string
            Recurrent Neural Network architecture to apply. Architectures LSTM, 
            GRU and SimpleRNN (alias sRNN) from Keras implementation are 
            available. From RNNNovelARchitectures module Fusion RNN (alias fRNN), 
            Neuro biomodulated recurrent cell (alias nBRC), 
            Biomodulated recurrent cell (alias BRC), Minimal gated unit (alias MGU) 
            and its variants (MGU1, MGU2 and MGU3).
        
        Returns
        -------
        model : tf.keras.utils.Sequence
            Recurrent Neural Network object based on the given settings.
    '''
    model = Sequential()
    input_shape = (X_shape[1], X_shape[2])
    
    novel_architecture = architecture not in ['LSTM', 'GRU', 'sRNN']
    
    for layer in range(n_layers):
        first_layer = layer == 0
        only_layer = n_layers == 1
        mid_layer = layer != 0 and layer != n_layers-1
        
        if only_layer: 
            args0 = {'input_shape':input_shape}
            print('Only layer, {}'.format(args0))
        elif first_layer:
            args0 = {'return_sequences':True,
                    'input_shape':input_shape}
            print('First layer, {}'.format(args0))
        elif mid_layer and not novel_architecture:
            args0 = {'return_sequences':True,
                     'dropout':dropout,
                     'recurrent_dropout':dropout}
            print('middle, {}'.format(args0))
        elif mid_layer and novel_architecture:
            args0 = {'return_sequences':True}
            print('middle, {} + in and out Dropout layers'.format(args0))
        else:
            args0 = {'return_sequences': False}
            print('last one, {}'.format(args0))

        print(f'{architecture}:', *args0)
        
        if novel_architecture:
            try:
                # Solution from: https://www.tutorialspoint.com/How-to-convert-a-string-to-a-Python-class-object
                architecture_object = getattr(sys.modules[__name__], architecture)
                rnn = RNN(architecture_object(units_layer), **args0)
            except TypeError:
                print('Unknown architecture.')            
        else:
            architecture_object = getattr(sys.modules[__name__], architecture)
            rnn = architecture_object(units_layer, **args0)
            
        if mid_layer and novel_architecture: 
            model.add(Dropout(dropout))
            model.add(rnn)
            model.add(Dropout(dropout))
        else:
            model.add(rnn)
        
    model.add(Dropout(dropout))
    
    print('y_shape: ', y_shape, '\tX_shape: ', X_shape )
    model.add(Dense(y_shape[1], activation='softmax'))

    return model

def History(h, wfolder, delnonoptimals=False):
    '''Function to load the file with the weight and biases that reach the early 
        stop criteria.
        
        Parameters
        ----------
        h : History object
            This object contains the Callbacks set in the Train Method of the 
            simpleLSTM class.
        
        Returns
        -------
        optimisedweights : string
            Name of the file with the weight and biases that fullfilled the 
            early stop criteria.
    '''
    cwd = getcwd()
    folder = r'weights'
    folderpath = join(cwd, folder, wfolder)
    loss = h.history['loss']
    magnitude = np.nanmin(loss)
    idx = np.where(loss == magnitude)[0][0]+1
    optimisedfile = 'weights-improvement-{:02}-{:.4f}.hdf5'.format(idx, magnitude)
    optimisedweights = join(folderpath, optimisedfile)
    
    if delnonoptimals:
        delfiles = [file for file in listdir(folderpath) if file != optimisedfile]
        for file in delfiles:
            remove(join(folderpath, file))
    
    return optimisedweights


def RNN_Iteration(seed_string='ABC', data_len=100, forecasting_len=10, X_windows=10, 
                  max_epochs=999, iterations=1, n_layers=1, units_per_layer=100, 
                  lr=0.01, batch_size=2**8, validation_split=0.05, 
                  architecture='LSTM', stop_param='loss', stop_val=0.1, 
                  del_nonoptimals=True, save_report=True, **kwargs):
    '''Function to produce one or several iterations of selected RNNs cells, with 
        different settings..
        
        Parameters
        ----------
        seed_string : str
            Seed-string of a determinate complexity to be learnt by the RNN.
        data_len : int
            Minimal length of dataset produced by the repetition of the 
            seed-string.
        max_epochs : int
            Upper limit of epochs to run during fitting.
        iterations : int
            Number of iterations to run with the given settings.
        n_layers : str
            Number of layers in the RNN cell.
        units_per_layer : str
            Number of units (cells) in each layer of the RNN.
        lr : float
            Initial learning rate for the adam optimizer.
        batch_size : int
            Size of the batch to process during fitting.
        validation_split : float
            Portion of training data to use as validation during fitting.
        architecture : str
            Type of cell, takes: LSTM, GRU, simple RNN, nBRC, BRC, MGU, MGU1
            MGU2 and MGU3.
        stop_param : str
            Alternative early stopping parameter, 'loss' for based on the 
            loss function, 'acc' for based on accuracy.
        stop_val : float
            Threshold that triggers the early stopping.
        del_nonoptimals :  boolean
            If True, all non optimal weights will be deleted.
        save_report : boolean
            If True a CSV file with a summary of the fitting is created.
        kwargs : dict
            Extra arguments used only to identify the seed-string.
        
        Returns
        -------
        df_summary : pandas.DataFrame
            Summary of the fitting process, with columns: jw (Jaro-Winkler 
            distance), dl (Damerau-Levenshtein distance), total_epochs 
            (epochs to reach stopping criterum), seq_test (validation 
            sequence), seq_forecast (forecasted sequence), total_time 
            (total time to reach stopping criteria). The rows are the 
            number of iterations.
    
    '''    
    
    print('Seed string: {}'.format(seed_string))
    sym_ts = SymbolicTS(seed_string, datalen=data_len)
    real_datalen = len(sym_ts)
    dataX, datay, dataTest = DataOneHotEncode(sym_ts,
                                             forecasting_len,
                                             X_windows)

    dataX = np.array(dataX)
    datay = np.array(datay)
    X_shape = np.shape(dataX)
    y_shape = np.shape(datay)
    print(X_shape, y_shape)
    
    df_summary = pd.DataFrame()
    for iteration_idx in range(iterations):
        
        M = simpleRNN(X_shape=X_shape, y_shape=y_shape, n_layers=n_layers, 
                       units_per_layer=units_per_layer, sym_ts=sym_ts, lr=lr, 
                      architecture=architecture)
        M.Build()
        M.Train(X=dataX, y=datay, epochs=max_epochs, batch_size=batch_size, 
                validation_split=validation_split, param=stop_param, stopval=stop_val)

        jw, dl, seq_test, seq_forecast = M.Forecast(dataY=datay, dataTest=dataTest, 
                                               delnonoptimals=del_nonoptimals)

        if save_report:

            df = pd.DataFrame(M.h.history)
            df['secs_epoch'] = M.process_time.times
            string_id = '{}'.format('.'.join('{:02}'.format(v) for v in kwargs.values())) if kwargs else 'NoInfo'
            name_features = [string_id, iteration_idx, n_layers, units_per_layer, architecture]
            df.to_csv('Monitoring-StringId-{}_Layers-{:02}_Cells-{:02}_{}.csv'.format(
                *name_features), index=False)

        total_epochs = len(M.h.history['loss'])
        total_time = sum(M.process_time.times)
        summary = dict(jw=jw, dl=dl, total_epochs=total_epochs, seq_test=seq_test, 
                       seq_forecast=seq_forecast, total_time=total_time)
        summary = pd.DataFrame(summary, index=[iteration_idx])
        df_summary = df_summary.append(summary)
    
    return df_summary 
