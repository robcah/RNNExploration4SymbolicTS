from tensorflow.keras.layers import Layer #, LSTM, GRU, RNN
from tensorflow.keras.initializers import constant, Orthogonal
from tensorflow.keras import optimizers
from tensorflow.nn import tanh, sigmoid, relu
from tensorflow import matmul, cast

from tensorflow import float32 as tf_float32
from tensorflow import zeros as tf_zeros




##### DEFINE THE nBRC ###########
class nBRC(Layer):
    # Neuromodulated Bistable Recurrent Cell (Vecoven et al., 2020)
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(nBRC, self).__init__(output_dim, **kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            })
        return config

    def build(self, input_shape):#, return_states):
        # U_c
        self.kernelz = self.add_weight(name="kz", shape=(input_shape[1], self.output_dim), 
                                      initializer='glorot_uniform')
        
        # U_a
        self.kernelr = self.add_weight(name="kr", shape=(input_shape[1], self.output_dim), 
                                      initializer='glorot_uniform')
        
        # U
        self.kernelh = self.add_weight(name="kh", shape=(input_shape[1], self.output_dim), 
                                      initializer='glorot_uniform')
        
        # W_c
        self.memoryz = self.add_weight(name="mz", shape=(self.output_dim, self.output_dim), 
                                      initializer='orthogonal')
        
        # W_a
        self.memoryr = self.add_weight(name="mr", shape=(self.output_dim, self.output_dim), 
                                      initializer='orthogonal')
        # Not specified
        self.br = self.add_weight(name="br", shape=(self.output_dim,),  
                                  initializer='zeros')
        
        # Not specified
        self.bz = self.add_weight(name="bz", shape=(self.output_dim,),  
                                  initializer='zeros')

        super(nBRC, self).build(input_shape)#, return_states)
    
    def call(self, inputs, states):
         # x_t
#         inp = inputs
        xt = cast(inputs, dtype='float32') 
        
        # h_{t-1}
#         prev_out = states[0]
        htm1 = cast(states[0], dtype='float32')
        
        # c_t
        z = sigmoid(matmul(xt, self.kernelz) + matmul(htm1, self.memoryz) + self.bz)
        
        # a_t
        r = tanh(matmul(xt, self.kernelr) + matmul(htm1, self.memoryr) + self.br)+1
        
        # h_t
        h = tanh(matmul(xt, self.kernelh) + r * htm1)
        output = (1.0 - z) * h + z * htm1
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf_float32):
        return [tf_zeros(shape=(batch_size, self.output_dim))]

############ DEFINE THE BRC #################
class BRC(Layer):
    # Bistable Recurrent Cell (Vecoven et al., 2020)
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(BRC, self).__init__(output_dim, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            })
        return config    
    
    def build(self, input_shape):#, return_sequences):
        #self.return_sequences = return_sequences
        # U_c
        self.kernelz = self.add_weight(name="kz", shape=(input_shape[1], self.output_dim), #dtype=tf_float32,
                                      initializer='glorot_uniform')
        # U_a
        self.kernelr = self.add_weight(name="kr", shape=(input_shape[1], self.output_dim), #dtype=tf_float32,
                                      initializer='glorot_uniform')
        # U
        self.kernelh = self.add_weight(name="kh", shape=(input_shape[1], self.output_dim), #dtype=tf_float32,
                                      initializer='glorot_uniform')
        
        # w_c
        self.memoryz = self.add_weight(name="mz", shape=(self.output_dim,), #dtype=tf_float32,
                        initializer=constant(1.0))
        
        # w_a
        self.memoryr = self.add_weight(name="mr", shape=(self.output_dim,), #dtype=tf_float32,
                        initializer=constant(1.0))

        # not specified in paper
        self.br = self.add_weight(name="br", shape=(self.output_dim,), #dtype = tf_float32, 
                                  initializer='zeros')
        
        # not specified in paper
        self.bz = self.add_weight(name="bz", shape=(self.output_dim,), #dtype = tf_float32, 
                                  initializer='zeros')

        super(BRC, self).build(input_shape)#, return_sequences)

    def call(self, inputs, states):
        # in the paper called x_t
#         inp = inputs
        xt = cast(inputs, dtype='float32')
        
        # in the paper called h_{t-1}
#         prev_out = states[0]
        htm1 = cast(states[0], dtype='float32')
        
        # in the paper called a_t
        r = tanh(matmul(xt, self.kernelr) + htm1 * self.memoryr + self.br) + 1
        
        # in the paper called c_t
        z = sigmoid(matmul(xt, self.kernelz) + htm1 * self.memoryz + self.bz)
        
        # in the paper called h_t
        output = z * htm1 + (1.0 - z) * tanh(matmul(xt, self.kernelh) + r * htm1)
        
#         if self.return_sequences:
#             output = outputs
#         else:
#             output = last_output
            
        return output, [output]


    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf_float32):
        return [tf_zeros(shape=(batch_size, self.output_dim))]
    
        
##### DEFINE THE MGU ###########
class MGU(Layer):
    # Minimal Gated Unit (Zhou et al., 2016)
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(MGU, self).__init__(output_dim, **kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            })
        return config

    def build(self, input_shape):
        self.Wf = self.add_weight(name="Wf", shape=(input_shape[1], self.output_dim), 
                                      initializer='glorot_uniform')
        self.Wh = self.add_weight(name="Wh", shape=(input_shape[1], self.output_dim), 
                                      initializer='glorot_uniform')

        self.Rf = self.add_weight(name="Rf", shape=(self.output_dim, self.output_dim), 
                                      initializer='orthogonal')
        self.Rh = self.add_weight(name="Rh", shape=(self.output_dim, self.output_dim), 
                                      initializer='orthogonal')

        self.Bh = self.add_weight(name="Bh", shape=(self.output_dim,), 
                                  initializer='zeros')
        self.Bf = self.add_weight(name="Bf", shape=(self.output_dim,), 
                                  initializer='zeros')
        
        super(MGU, self).build(input_shape)
    
    def call(self, inputs, states, training=None):
#         inp = inputs
        xt = cast(inputs, dtype='float32') 
        
#         prev_out = states[0]
        htm1 = cast(states[0], dtype='float32')
        ft = sigmoid(matmul(xt, self.Wf) + matmul(htm1, self.Rf) + self.Bf)
        dot_ht = tanh(matmul(xt, self.Wh) + matmul((ft * htm1), self.Rh) + self.Bh) 
        output = (1.0 - ft) * htm1 + ft * dot_ht
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf_float32):
        return [tf_zeros(shape=(batch_size, self.output_dim))]

    
##### DEFINE THE MGU1 ###########
class MGU1(Layer):
    # Minimal Gated Unit - simplified version 1 (Heck et al., 2017)
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(MGU1, self).__init__(output_dim, **kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            })
        return config

    def build(self, input_shape):
        self.Wh = self.add_weight(name="Wh", shape=(input_shape[1], self.output_dim), 
                                      initializer='glorot_uniform')

        self.Rf = self.add_weight(name="Rf", shape=(self.output_dim, self.output_dim), 
                                      initializer='orthogonal')
        self.Rh = self.add_weight(name="Rh", shape=(self.output_dim, self.output_dim), 
                                      initializer='orthogonal')

        self.Bh = self.add_weight(name="Bh", shape=(self.output_dim,), 
                                  initializer='zeros')
        self.Bf = self.add_weight(name="Bf", shape=(self.output_dim,), 
                                  initializer='zeros')
        super(MGU1, self).build(input_shape)
    
    def call(self, inputs, states, training=None):
#         inp = inputs
        xt = cast(inputs, dtype='float32') 
        
#         prev_out = states[0]
        htm1 = cast(states[0], dtype='float32')
        ft = sigmoid(matmul(htm1, self.Rf) + self.Bf)
        dot_ht = tanh(matmul(xt, self.Wh) + matmul((ft * htm1), self.Rh) + self.Bh) 
        output = (1.0 - ft) * htm1 + ft * dot_ht
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf_float32):
        return [tf_zeros(shape=(batch_size, self.output_dim))]

    
    ##### DEFINE THE MGU2 ###########
class MGU2(Layer):
    # Minimal Gated Unit - simplified version 2 (Heck et al., 2017)
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(MGU2, self).__init__(output_dim, **kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            })
        return config

    def build(self, input_shape):
        self.Wh = self.add_weight(name="Wh", shape=(input_shape[1], self.output_dim), 
                                      initializer='glorot_uniform')

        self.Rf = self.add_weight(name="Rf", shape=(self.output_dim, self.output_dim), 
                                      initializer='orthogonal')
        self.Rh = self.add_weight(name="Rh", shape=(self.output_dim, self.output_dim), 
                                      initializer='orthogonal')

        self.Bh = self.add_weight(name="Bh", shape=(self.output_dim,), 
                                  initializer='zeros')
        super(MGU2, self).build(input_shape)
    
    def call(self, inputs, states, training=None):
#         inp = inputs
        xt = cast(inputs, dtype='float32') 
        
#         prev_out = states[0]
        htm1 = cast(states[0], dtype='float32')
        ft = sigmoid(matmul(htm1, self.Rf))
        dot_ht = tanh(matmul(xt, self.Wh) + matmul((ft * htm1), self.Rh) + self.Bh) 
        output = (1.0 - ft) * htm1 + ft * dot_ht
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf_float32):
        return [tf_zeros(shape=(batch_size, self.output_dim))]


##### DEFINE THE MGU3 ###########
class MGU3(Layer):
    # Minimal Gated Unit - simplified version 3 (Heck et al., 2017)
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(MGU3, self).__init__(output_dim, **kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            })
        return config

    def build(self, input_shape):
        self.Wh = self.add_weight(name="Wh", shape=(input_shape[1], self.output_dim), 
                                      initializer='glorot_uniform')
        self.Rh = self.add_weight(name="Rh", shape=(self.output_dim, self.output_dim), 
                                      initializer='orthogonal')

        self.Bh = self.add_weight(name="Bh", shape=(self.output_dim,), 
                                  initializer='zeros')
        self.Bf = self.add_weight(name="Bf", shape=(self.output_dim,), 
                                  initializer='zeros')
        
        super(MGU3, self).build(input_shape)
    
    def call(self, inputs, states, training=None):
#         inp = inputs
        xt = cast(inputs, dtype='float32') 
        
#         prev_out = states[0]
        htm1 = cast(states[0], dtype='float32')
        ft = sigmoid(self.Bf)
        dot_ht = tanh(matmul(xt, self.Wh) + matmul((ft * htm1), self.Rh) + self.Bh) 
        output = (1.0 - ft) * htm1 + ft * dot_ht
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf_float32):
        return [tf_zeros(shape=(batch_size, self.output_dim))]


##### DEFINE THE fRNN ###########
class fRNN(Layer):
    # Fusion Recurrent Neural Network (Sun et al., 2020)
    def __init__(self, output_dim, r=5, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        self.r = r
        super(fRNN, self).__init__(output_dim, **kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            })
        return config

    def build(self, input_shape):
        self.Fx = self.add_weight(name="Fx", shape=(self.output_dim, input_shape[1]), 
                                      initializer='glorot_uniform')
        self.Fh = self.add_weight(name="Fh", shape=(input_shape[1], self.output_dim), 
                                      initializer='orthogonal')
        
        self.Wx = self.add_weight(name="Wx", shape=(input_shape[1], self.output_dim), 
                                      initializer='glorot_uniform')
        self.Wh = self.add_weight(name="Wh", shape=(self.output_dim, self.output_dim), 
                                      initializer='orthogonal')

        self.b = self.add_weight(name="b", shape=(self.output_dim,),  
                                  initializer='zeros')
        
        super(fRNN, self).build(input_shape)
    
    def call(self, inputs, states, training=None):
        xt = cast(inputs, dtype='float32')
        htm1 = cast(states[0], dtype='float32')
        for i in range(self.r+1):
            first = i == 0
            even = i % 2 == 0
            if first:
                xti = sigmoid(matmul(htm1, self.Fx) * xt)
                htm1i = htm1
            if even and not first:
                htm1i = sigmoid(matmul(xti, self.Fh) * htm1i)
            else:
                xti = sigmoid(matmul(htm1i, self.Fx) * xti)
        output = tanh(matmul(xti, self.Wx) + matmul(htm1i, self.Wh) + self.b)
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf_float32):
        return [tf_zeros(shape=(batch_size, self.output_dim))]


class GORU(Layer):
    # Gated Orthogonal Recurrent Unit (Jing et al., 2017)
    # Orthogonal matrix Uh requires more work
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.state_size = output_dim
        super(GORU, self).__init__(output_dim, **kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            })
        return config

    def build(self, input_shape):
        self.Wz = self.add_weight(name="Wz", shape=(input_shape[1], self.output_dim), 
                                      initializer='glorot_uniform')
        self.Wr = self.add_weight(name="Wr", shape=(input_shape[1], self.output_dim), 
                                      initializer='glorot_uniform')
        self.Wx = self.add_weight(name="Wx", shape=(input_shape[1], self.output_dim), 
                                      initializer='glorot_uniform')
        self.Uh = self.add_weight(name="Uh", shape=(self.output_dim, self.output_dim), 
                                      initializer=Orthogonal())

        self.Rz = self.add_weight(name="Rz", shape=(self.output_dim, self.output_dim), 
                                      initializer='orthogonal')
        self.Rr = self.add_weight(name="Rr", shape=(self.output_dim, self.output_dim), 
                                      initializer='orthogonal')

        self.Bz = self.add_weight(name="Bz", shape=(self.output_dim,), 
                                  initializer='zeros')
        self.Br = self.add_weight(name="Br", shape=(self.output_dim,), 
                                  initializer='zeros')
        self.Bh = self.add_weight(name="Bh", shape=(self.output_dim,), 
                                  initializer='zeros')
        
        super(GORU, self).build(input_shape)
    
        
    @staticmethod
    def modReLU(zi, bi):
        return (zi/abs(zi)) * relu(abs(zi) + bi)
    
    def call(self, inputs, states, training=None):
#         inp = inputs
        xt = cast(inputs, dtype='float32') 
#         prev_out = states[0]
        htm1 = cast(states[0], dtype='float32')
    
        zt = sigmoid(matmul(xt, self.Wz) + matmul(htm1, self.Rz) + self.Bz)
        rt = sigmoid(matmul(xt, self.Wr) + matmul(htm1, self.Rr) + self.Br)
        zi = matmul(xt, self.Wx) + rt * matmul(htm1, self.Uh)
        output = zt * htm1 + (1.0 - zt) * self.modReLU(zi, self.Bh)
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf_float32):
        return [tf_zeros(shape=(batch_size, self.output_dim))]