import numpy as np
import tensorflow as tf

def _get_norm_layer(norm, *args, **kwargs):
    if norm=="lrn":
        return lambda x: tf.nn.lrn(x,
                         depth_radius=7,
                         alpha=0.001,
                         beta=0.75)
    if norm=="bn":
        return tf.keras.layers.BatchNormalization(name=kwargs["name"])

class RecurrentConvLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, timestep=3, norm_method="bn", weight_decay=0.0):
        super(RecurrentConvLayer, self).__init__()
        self.num_outputs = num_outputs
        self.T = 3
        self.norm_method = norm_method
        self.weight_decay = weight_decay

    def build(self, input_shape):
        self.convBx = self.add_variable("convBx",                                          
                                    [self.num_outputs], 
                                    trainable=True,
                                    initializer=tf.keras.initializers.constant(0))
        #Shared weights for recurrent connections
        self.convWh = self.add_weight("convWh",
                                    [3,3,self.num_outputs,self.num_outputs],
                                    trainable=True,
                                    initializer=tf.keras.initializers.glorot_normal())
        self.conv1 = tf.keras.layers.Conv2D(self.num_outputs,
                                            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                            kernel_size=(3,3),
                                            padding="SAME",
                                            activation="linear",
                                            use_bias=False)
        self.norm1 = _get_norm_layer(self.norm_method, name = "normLayer_1")
        self.rec_norm = []
        self.rconv_layers = []
        for i in range(1,self.T):
            self.rec_norm.append(_get_norm_layer(self.norm_method, name="normLayer_rec_{}".format(i)))
            conv_layer = tf.keras.layers.Conv2D(self.num_outputs, 
                                                kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                                kernel_size=(3,3),
                                                padding="SAME",
                                                activation="linear",
                                                use_bias=False,
                                                name="RCL_{}".format(i))
            conv_layer.build(input_shape)
            self.rconv_layers.append(conv_layer)
        super(RecurrentConvLayer, self).build(input_shape)
        # All the recurrent connections share the same weight
        for i in range(len(self.rconv_layers)):
            self.rconv_layers[i].kernel = self.convWh
            self.rconv_layers[i]._trainable_weights = []
            self.rconv_layers[i]._trainable_weights.append(self.convWh)
            print("Layer ", i, self.rconv_layers[i]._trainable_weights)
        self.build=True


    def call(self,input):
        # t=0 : a feedforward pass
        x = self.conv1(input)
        bn1 = self.norm1(x)
        nonlinear = tf.keras.layers.ReLU()(bn1)

        for i in range(1,self.T):
            rec = self.rconv_layers[i-1](nonlinear) #tf.keras.backend.conv2d(nonlinear,
                  #                        self.convWh,
                  #                        padding="same")
            s = tf.add(x, rec)
            #add normalization
            s = self.rec_norm[i-1](s)
            if i == self.T-1:
                s = tf.add(s, self.convBx)
            nonlinear = tf.keras.layers.ReLU()(s)
        return nonlinear

class Model:
    def __init__(self, filer_size, keep_prob, numclass):
        self.K = filer_size
        self.p = keep_prob
        self.numclass = numclass
    def call(self, input_shape):
        pass

class RCNN(Model):
    def __init__(self,
                 time,
                 filer_size,
                 keep_prob,
                 numclass,
                 weight_decay=0.0,
                 norm_method="bn",
                 nrof_rec_layers=5):

        super(RCNN, self).__init__(filer_size, keep_prob, numclass)
        self.time = time
        self.nrof_rec_layers = nrof_rec_layers
        self.norm_method = norm_method
        self.weight_decay = weight_decay

    def call(self, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape)
        conv1 = tf.keras.layers.Conv2D(self.K,
                                       kernel_size= (5, 5),
                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                       padding="SAME",
                                       use_bias=True)(inputs)
        output = tf.keras.layers.BatchNormalization()(conv1)
        output = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='VALID')(output)

        for i in range(1,self.nrof_rec_layers):
            output = RecurrentConvLayer(self.K,
                                        self.time,
                                        self.norm_method,
                                        weight_decay = self.weight_decay)(output)
            if (i%2==0) and (i!=self.nrof_rec_layers-1) :
                output = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='VALID')(output)
            if (i!=self.nrof_rec_layers-1):
                output = tf.keras.layers.Dropout(1-self.p)(output)

        output = tf.keras.layers.GlobalMaxPool2D()(output)   
        logits = tf.keras.layers.Dense(self.numclass,
                                        activation="softmax",
                                        use_bias=True)(output)
        return tf.keras.Model(inputs=inputs, outputs=logits,name="RCNN")

class WCNN(Model):
    def __init__(self, filer_size, keep_prob, numclass, weight_decay = 0.0, numlayers=5):
        super(WCNN, self).__init__(filer_size, keep_prob, numclass) 
        self.numlayers = numlayers
        self.weight_decay = weight_decay
    def call(self, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape)
        output = tf.keras.layers.Conv2D(self.K,
                                       kernel_size= (5, 5),
                                       padding="SAME",
                                       kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                       use_bias=True)(inputs)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='VALID')(output)
        for i in range(1,self.numlayers):
            output = tf.keras.layers.Conv2D(self.K,
                                            kernel_size=(3,3),
                                            kernel_regularizer=tf.keras.regularizers.l2(self.weight_decay),
                                            padding="SAME",
                                            use_bias=True)(output)
            output = tf.keras.layers.BatchNormalization()(output)
            if (i%2==0) and (i!=self.numlayers-1):
                output = tf.keras.layers.MaxPool2D(pool_size=2,
                                                   strides=2,
                                                   padding='VALID')(output)
            if i != self.numlayers-1:
                output = tf.keras.layers.SpatialDropout2D(1-self.p)(output)
        output = tf.keras.layers.GlobalMaxPool2D()(output)
        logits = tf.keras.layers.Dense(self.numclass,
                                       activation="softmax",
                                       use_bias=True)(output)
        return tf.keras.Model(inputs=inputs, outputs=logits,name="WCNN")
