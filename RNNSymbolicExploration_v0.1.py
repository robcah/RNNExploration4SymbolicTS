# !python3 -m pip install textdistance

import sys
import numpy as np
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import optimizers
import pandas as pd
from pathlib import Path
from os.path import join, exists
from os import listdir, getcwd
import textdistance as td
import time as T

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
            
    def Build(self):
        '''Calls for the model building with the settings stated in the class, 
        compiles it with Adam optimiser and categorical crossentropy as loss function.        
        '''
        adam = optimizers.Adam(learning_rate=lr)
        clear_session()
        self.model = Model(self.X_shape, self.y_shape, self.n_layers, self.units_per_layer, 
                           self.dropout, self.architecture)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
            
#     @staticmethod
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
        filepath = join(cwd, folder, r'weights-improvement-{epoch:02d}-{loss:.4f}.hdf5')
        
        def checkpoint_def():
            return ModelCheckpoint(filepath, monitor='loss', 
                verbose=1, save_best_only=True, mode='min')
        
        # Try to find folder otherwise it creates it
        if not exists(join(cwd, folder)):
            Path(join(cwd, folder)).mkdir(parents=True, exist_ok=True)
        
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
    
    # batch_size: the number of training examples in one forward/backward pass.
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
        # here goes the early stop param and stopval
        
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
        
    def Forecast(self, model=None, dataY=None, dataTest=None, filename=None):
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
        # load the data and define the network in exactly the same way, 
        # except the network weights are loaded from a checkpoint file 
        # and the network does not need to be trained.

        if model is None: model = self.model
        if dataY is None and dataTest is None: 
            _, dataY, dataTest = data.Data()

            # this function doesn't use dataY!!!
            # pattern is the seed, namely the chunk from 
            # which the forecast is made, therefore using dataY might be as good
        seq_test = ''.join(dataTest) # this should be dataTest, OK!
        
        filename = History(self.h)
        print(f'Using: {filename}')

        n_vocab = len(SymbolList(self.sym_ts)) # Check this!! connect to dynamic flow
        model.load_weights(filename)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        # reverse mapping that we can use to convert the integers back to characters
        int_to_char = SymbolList(self.sym_ts)

        # make prediction
        pattern = dataY[-100:]
        print("Seed:")
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
            Recurrent Neural Network.
        
        Returns
        -------
        model : tf.keras.utils.Sequence
            Recurrent Neural Network object based on the given settings.
    '''
    model = Sequential()
    input_shape = (X_shape[1], X_shape[2])
    
    for layer in range(n_layers):
        if layer == 0:
            print('First layer')
            args0 = {'return_sequences': True,
                    'input_shape': input_shape}
            if n_layers == 1: 
                print('Only layer')
                args0 = {'input_shape': input_shape}
        elif layer != 0 and layer != n_layers-1:
            print('middle')
            args0 = {'return_sequences': True, 'dropout': dropout,
                   'recurrent_dropout': dropout}
        else:
            print('last one')
            args0 = {'return_sequences': False}

        print(f'{architecture}:', *args0)
        if architecture == 'LSTM':
            RNN = LSTM
        elif architecture == 'GRU':
            RNN = GRU
        else:
            raise Exception("Architecture unknown.")
        model.add(RNN(units_layer, **args0)) 
        model.add(Dropout(dropout))
    
    model.add(Dense(y_shape[1], activation='softmax'))

    return model

def History(h):
    '''Function to load the file with the weight and biases that reach the early 
        stop criteria.
        
        Parameters
        ----------
        h : History object
            This object contains the Callbacks set in the Train Method of the 
            simpleLSTM class.
        
        Returns
        -------
        filename : string
            Name of the file with the weight and biases that fillfilled the 
            early stop criteria.
    '''
    cwd = getcwd()
    folder = r'weights'
    loss = h.history['loss']
    magnitude = np.nanmin(loss)
    idx = np.where(loss == magnitude)[0][0]+1
    filename = join(cwd, folder, 'weights-improvement-{:02}-{:.4f}.hdf5'.format(idx, magnitude))
    
    return filename
    
    def RNN_Iteration(i, j, complexity, symbols, string, n_layers, units_per_layer, 
                   epochs, batch_size, validation_split, lr, datalen, architecture):
    
    print(string)
    sym_ts = SymbolicTS(string, datalen=datalen)
    dataX, datay, dataTest = DataOneHotEncode(sym_ts)

    dataX = np.array(dataX)
    datay = np.array(datay)
    X_shape = np.shape(dataX)
    y_shape = np.shape(datay)
    print(X_shape, y_shape)

    M = simpleRNN(X_shape=X_shape, y_shape=y_shape, n_layers=n_layers, 
                   units_per_layer=units_per_layer, sym_ts=sym_ts, lr=lr, 
                  architecture=architecture)
    M.Build()
    M.Train(X=dataX, y=datay, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
            param='acc', stopval=0.99)
    
    jw, dl, seq_test, seq_out = M.Forecast(dataY=datay, dataTest=dataTest)
    
    total_eps = len(M.h.history['loss'])
    
    df = pd.DataFrame(M.h.history)
    df['secs_epoch'] = M.process_time.times
    df.to_csv('File-{:02}.{:02}_C{:02}S{:02}_Layers-{:02}_Cells-{:02}_monitor.csv'.format(
                i, j, complexity, symbols, n_layers, units_per_layer), index=False)
    total_time = sum(M.process_time.times)
    
    return jw, dl, total_eps, seq_test, seq_out, total_time, architecture