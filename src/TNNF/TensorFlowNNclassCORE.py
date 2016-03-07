from PIL import Image, ImageOps, ImageFilter
import numpy as np
from numpy import *
from numpy import dot, sqrt, diag
from numpy.linalg import eigh
import cPickle
import time  # import datetime
from fImageWorkerCORE import *
import tensorflow as tf
import sys

#---------------------------------------------------------------------#
# Graph Function TensorFlow
#---------------------------------------------------------------------#


class TensorFlowFunction(object):
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs
    def __call__(self, *args, **kwargs):
        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg
        return tf.get_default_session().run(self._outputs, feeds)

#---------------------------------------------------------------------#
# Activation functions
#---------------------------------------------------------------------#


class FunctionModel(object):
    @staticmethod  # FunctionModel.Sigmoid
    #def Sigmoid(W, X, B, *args):
    def Sigmoid(z, *args):
        #a = 1 / (1 + tf.exp(-z))
        a = tf.nn.sigmoid(z)
        return a

    @staticmethod  # FunctionModel.RelU
    #def ReLU(W, X, B, *args):
    def ReLU(z, *args):
        # a = tf.cond(tf.greater(z, 0), z, 0)
        a = tf.nn.relu(z)
        return a

    @staticmethod  # FunctionModel.LReLU
    #def LReLU(W, X, B, *args):
    def LReLU(z, *args):
        a = tf.cond(tf.greater(z, 0), z, z * 0.01)
        return a

    @staticmethod  # FunctionModel.Linear
    #def Linear(W, X, B, *args):
    def Linear(z, *args):
        return z

    @staticmethod  # FunctionModel.Tanh
    #def Tanh(W, X, B, *args):
    def Tanh(z, *args):
        #a = (tf.exp(z) - tf.exp(-z)) / (tf.exp(z) + tf.exp(-z))
        a = tf.tanh(z)
        return z

    @staticmethod  # FunctionModel.SoftMax
    #def SoftMax(W, X, B, *args):
    def SoftMax(z, *args):
        a = tf.nn.softmax(z)
        #numClasses = tf.shape(z)[0]
        #z_max = tf.reduce_max(z, reduction_indices=0, keep_dims=False)
        #a = tf.exp(z - tf.log(tf.matmul(tf.constant(1.0, shape=[numClasses, 1]), [tf.reduce_sum(tf.exp(z-z_max), reduction_indices=0)])))
        return a

    @staticmethod  # FunctionModel.MaxOut
    #def MaxOut(W, X, B, *args):
    def MaxOut(z, *args):
        d = tf.shape(z)
        n_elem = args[0]
        z = tf.reshape(z, [d[0] / n_elem, n_elem, d[1]])
        a = tf.reduce_max(z, reduction_indices=1, keep_dims=False)
        return a

# ---------------------------------------------------------------------#
# Layer builders
# ---------------------------------------------------------------------#

class LayerNN(object):
    def __init__(self,
                 size_in=1,
                 size_out=1,
                 activation=FunctionModel.Sigmoid,
                 weightDecay=False,
                 sparsity=False,
                 beta=False,
                 dropout=False,
                 dropConnect=False,
                 pool_size=False):

        self.size_in = size_in
        self.size_out = size_out
        self.activation = activation
        self.weightDecay = weightDecay
        self.sparsity = sparsity
        self.beta = beta
        self.dropout = dropout
        self.dropConnect = dropConnect
        self.pool_size = pool_size

    def Printer(self):
        print self.__dict__

    def compileWeight(self, net, layerNum):
        random = sqrt(6) / sqrt(net.architecture[0].size_in + net.architecture[-1].size_out)
        W = dict()

        #if self.activation != FunctionModel.MaxOut:
        #    weights = np.random.randn(self.size_out, self.size_in)
        #else:
        #    weights = np.random.randn(self.size_out * self.pool_size, self.size_in)

        #print("Weights")
        #print(weights)
        #w = tf.get_variable(name="w%s" % (layerNum + 1), shape=weights.shape, dtype=tf.float32, initializer=tf.constant_initializer((weights * 0.01)))


        if self.activation != FunctionModel.MaxOut:
            w = tf.Variable(tf.random_normal([self.size_out, self.size_in], stddev=0.01), name="w%s" % (layerNum + 1))
        else:
            w = tf.Variable(tf.random_normal([self.size_out * self.pool_size, self.size_in], stddev=0.01), name="w%s" % (layerNum + 1))

        W['w'] = w
        if self.activation != FunctionModel.MaxOut:
            b = tf.Variable(np.tile(0.1, (self.size_out,)).astype("float32"), name="b%s" % (layerNum + 1))
        else:
            b = tf.Variable(np.tile(0.1, (self.size_out * self.pool_size,)).astype("float32"), name="b%s" % (layerNum + 1))
        W['b'] = b

        #print "NN-Compile Weight"
        #model = tf.initialize_all_variables()
        #with tf.Session() as session:
        #    session.run(model)
        #    print(session.run(W['w']))

        net.varWeights.append(W)

    def compileDropout(self, net):
        #Assume we work only with square kernels
        #if needed add noise_shape parameter with (self.size_in,)
        if self.dropout:
            net.dropOutVectors.append(tf.nn.dropout(self.size_in * 1., keep_prob=self.dropout))
        else:
            net.dropOutVectors.append(1.0)

    def compileActivation(self, net, layerNum):
        variable = net.x if layerNum == 0 else net.varArrayA[layerNum - 1]
        #print tf.expand_dims(net.dropOutVectors[layerNum], 1)

        #print net.varWeights[layerNum]['w'].get_shape().as_list()
        print "LayerCNN - Compile Activation Start"
        print(layerNum)

        #variable = tf.placeholder("int")
        print "LayerCNN - Compile Activation End"
        #z = tf.reduce_prod((net.varWeights[layerNum]['w'])) + tf.reshape(net.varWeights[layerNum]['b'], [net.varWeights[layerNum]['w'].get_shape().as_list()[0],1])
        z = 128
        a = self.activation(z, self.pool_size)
        net.varArrayA.append(a)

    def compilePredictActivation(self, net, layerNum):
        variable = net.x if layerNum == 0 else net.varArrayA[layerNum - 1]
        #z = tf.matmul(net.varWeights[layerNum]['w'] * (self.dropout if self.dropout else 1.0), tf.reshape(variable, [1, net.varWeights[layerNum]['w'].get_shape().as_list()[0]*2])) + tf.expand_dims(net.varWeights[layerNum]['b'], 1)
        z = 128
        a = self.activation(z, self.pool_size)
        net.varArrayA.append(a)

    def compileSparsity(self, net, layerNum, num):
        sprs = tf.reduce_sum(net.varArrayA[layerNum], reduction_indices=1) / (num + 0.0)
        epsilon = 1e-20
        sprs = tf.clip_by_value(sprs, epsilon, 1 - epsilon)
        KL = T.reduce_sum(
            self.sparsity * tf.log(self.sparsity / sprs) + (1 - self.sparsity) * tf.log((1 - self.sparsity) / (1 - sprs)))
        net.regularize.append(self.beta * KL)

    def compileWeightDecayPenalty(self, net, layerNum):
        print self.weightDecay / 2
        penalty = T.reduce_sum(tf.pow(net.varWeights[layerNum]['w'],2)) * self.weightDecay / 2
        net.regularize.append(penalty)

class LayerCNN(LayerNN):
    def __init__(self, kernel_shape=None, stride=1, pooling=False, pooling_shape=None, optimized=False, **kwargs):

        super(LayerCNN, self).__init__(**kwargs)

        self.kernel_shape = kernel_shape        #number of kernels
        self.stride = stride                    #stride for convolution
        self.pooling = pooling                  #Pooling ON|OFF
        self.pooling_shape  = pooling_shape     #Pooling strides
        self.optimized = optimized

    def compileWeight(self, net, layerNum):
        random = sqrt(6) / sqrt(self.kernel_shape[-1] * self.kernel_shape[-2] * self.kernel_shape[0])
        W = dict()

        if self.activation == FunctionModel.MaxOut:
            raise NotImplementedError('MaxOut activation function for Convolution nets is not implemented yet!')

        #weights = np.random.randn(*self.kernel_shape)
        w = tf.Variable(tf.random_normal(self.kernel_shape, stddev=0.01), name="w%s" % (layerNum + 1))
        W['w'] = w

        #bias shape == number of kernels
        b = tf.Variable(np.tile(0.1, (self.kernel_shape[0],)).astype("float32"), name="b%s" % (layerNum + 1))
        W['b'] = b
        #print "CNN-Compile Weight"
        #model = tf.initialize_all_variables()
        #with tf.Session() as session:
        #    session.run(model)
        #    print(session.run(W['w']))
        net.varWeights.append(W)

    def compileDropout(self, net):
        #Assume we work only with square kernels
        #if needed add noise_shape parameter with (self.kernel_shape[-2], self.kernel_shape[-1])
        if self.dropout:
            net.dropOutVectors.append(tf.nn.dropout(self.kernel_shape * 1., keep_prob=self.dropout))
        else:
            net.dropOutVectors.append(1.0)

    def compileActivation(self, net, layerNum):
        variable = net.x if layerNum == 0 else net.varArrayA[layerNum - 1]

        print "LayerCNN - Compile Activation Start"
        print(layerNum)
        if layerNum != 0:
            print(net.varArrayA[layerNum - 1])
        print(variable.get_shape())
        print(tf.shape(variable)[0])
        print "LayerCNN - Compile Activation End"
        '''
        sh = tf.shape(variable)
        sess = tf.Session()
        shap_var =  sess.run(sh, feed_dict={net.x: [12.4, 14.54, 88.43,13.5,99.23, 23.21]})
        '''
        #Calc shapes for reshape function on-the-fly. Assume we have square images as input.
        sX = tf.cast(tf.sqrt(tf.shape(variable)[0] / self.kernel_shape[1]), tf.int32)

        #Converts input from 2 to 4 dimensions
        variable = tf.transpose(variable)

        #sess = tf.Session()
        #print sess.run(sh, feed_dict={net.x: [12.4, 14.54, 88.43, 13.5]})


        #print(variable, tf.shape(variable)[1], self.kernel_shape[1], sX, sX.eval())
        Xr = tf.cast(tf.reshape(variable, tf.pack([tf.shape(variable)[0], self.kernel_shape[1], sX, sX])), tf.float32)

        #tensorFlow conv2d works only with float32 if required cast Xr to float32
        #The dimesion ordering of tensorflow is different for both Xr and filterr
        Xr = tf.transpose(Xr, (0, 2, 3, 1))


        filterr = net.varWeights[layerNum]['w'] * (tf.reshape(net.dropOutVectors[layerNum],(1, 1, net.dropOutVectors[layerNum].shape[0], net.dropOutVectors[layerNum].shape[1])) if self.dropout else 1.0)


        filterr = tf.transpose(filterr, (2, 3, 1, 0))

        ##left from here
        if self.optimized:
            a = tf.nn.conv2d(Xr, filter=filterr, strides=[1, self.stride,self.stride,1], padding='VALID')
        else:
            a = tf.nn.conv2d(Xr, filter=filterr, strides=[1, self.stride,self.stride,1], padding='VALID', use_cudnn_on_gpu=False)


        #Add Bias
        a = a + tf.reshape(net.varWeights[layerNum]['b'], (1, net.varWeights[layerNum]['b'].get_shape().as_list()[0], 1, 1))
        print a
        #Max pooling in tensorflow requires float32 input if required cast a to float32
        if self.pooling:
            a =  tf.transpose(a, (0, 2, 3, 1))
            a = tf.nn.max_pool(a, ksize=[1, self.pooling_shape, self.pooling_shape, 1], strides=[1,self.pooling_shape,self.pooling_shape ,1], padding='VALID')
            print "POOL"
        else:
            if self.optimized:
                a =  tf.transpose(a, (0, 2, 3, 1))
        a = tf.reshape(a, [-2])
        a = tf.transpose(a)

        a = self.activation(a, self.pool_size)

        net.varArrayA.append(a)


    def compileSparsity(self, net, layerNum, num):
        a = net.varArrayA[layerNum]
        out_size = tf.cast(tf.sqrt(tf.shape(a)[0] / self.kernel_shape[0]), tf.int16)
        a = tf.reshape(a, (net.options.minibatch_size, self.kernel_shape[0], out_size, out_size))
        sprs = tf.reduce_mean(a, reduction_indices=1)
        epsilon = 1e-20
        sprs = tf.clip_by_value(sprs, epsilon, 1 - epsilon)
        KL = tf.reduce_sum(self.sparsity * tf.log(self.sparsity / sprs) + (1 - self.sparsity) * tf.log((1 - self.sparsity) / (1 - sprs))) / (out_size * out_size)
        net.regularize.append(self.beta * KL)

    def compilePredictActivation(self, net, layerNum):
        variable = net.x if layerNum == 0 else net.varArrayAc[layerNum - 1]

        #Calc shapes for reshape function on-the-fly. Assume we have square images as input.
        sX = tf.cast(tf.sqrt(tf.shape(variable)[0] / self.kernel_shape[1]), tf.int32)

        #Converts input from 2 to 4 dimensions
        variable = tf.transpose(variable)

        #sess = tf.Session()
        #print sess.run(sh, feed_dict={net.x: [12.4, 14.54, 88.43, 13.5]})


        #print(variable, tf.shape(variable)[1], self.kernel_shape[1], sX, sX.eval())
        Xr = tf.cast(tf.reshape(variable, tf.pack([tf.shape(variable)[0], self.kernel_shape[1], sX, sX])), tf.float32)
        Xr = tf.transpose(Xr, (0, 2, 3, 1))


        filterr = net.varWeights[layerNum]['w'] * (tf.reshape(net.dropOutVectors[layerNum],(1, 1, net.dropOutVectors[layerNum].shape[0], net.dropOutVectors[layerNum].shape[1])) if self.dropout else 1.0)


        filterr = tf.transpose(filterr, (2, 3, 1, 0))
        if self.optimized:
            a = tf.nn.conv2d(Xr, filter=filterr, strides=[1, self.stride,self.stride,1], padding='VALID')
        else:
            a = tf.nn.conv2d(Xr, filter=filterr, strides=[1, self.stride,self.stride,1], padding='VALID', use_cudnn_on_gpu=False)

        a = a + tf.reshape(net.varWeights[layerNum]['b'], (1, net.varWeights[layerNum]['b'].get_shape().as_list()[0], 1, 1))

        if self.pooling:
            a =  tf.transpose(a, (0, 2, 3, 1))
            a = tf.nn.max_pool(a, ksize=[1, self.pooling_shape, self.pooling_shape, 1], strides=[1,self.pooling_shape,self.pooling_shape ,1], padding='VALID')
            #print "POOL"
        else:
            if self.optimized:
                a =  tf.transpose(a, (0, 2, 3, 1))
        a = tf.reshape(a, [-2])
        a = tf.transpose(a)

        a = self.activation(a, self.pool_size)

        net.varArrayAc.append(a)

#---------------------------------------------------------------------#
# Basic neural net class
#---------------------------------------------------------------------#

class TensorFlowNNclass(object):
    def __init__(self, opt, architecture):
        self.REPORT = "OK"
        self.architecture = architecture
        self.options = opt
        self.lastArrayNum = len(architecture)

        self.varWeights = []

        #variables
        self.x = tf.placeholder("float64", name="x")
        self.y = tf.placeholder("float64", name="y")
        model = tf.initialize_all_variables()



        #Initialize Weights
        for i in xrange(self.lastArrayNum):
            self.architecture[i].compileWeight(self, i)

        # Dropout
        self.dropOutVectors = []
        for i in xrange(self.lastArrayNum):
            self.architecture[i].compileDropout(self)

        #Activation list
        self.varArrayA = []

        #Additional penalty list
        self.regularize = []

        #Error Calculation
        self.errorArray = []  # Storage for costs
        self.cost = 0

        #Derivatives array
        self.derivativesArray = []

        # RMS
        if self.options.rmsProp:
            self.MMSprev = []
            self.MMSnew = []

        # Update array
        self.updatesArray = [] #currently not required in tensorflow; lets see

        # Sometimes there is something to update even for predict (say in RNN)
        self.updatesArrayPredict = []

        # train
        self.train = None
        self.trainExternal = None

        # predict
        self.predict = None
        self.out = None

        #For external train
        self.metadata = None
        self.unrolledModel = None
        self.unroll()

        #Predict Variables
        self.data = tf.placeholder("float64", name="data")
        self.varArrayAc = []

        # List of output variables
        self.outputArray = []

    def trainCompile(self):

        # Activation
        for i in xrange(self.lastArrayNum):
            self.architecture[i].compileActivation(self, i)

        # Sparse penalty
        for i in xrange(self.lastArrayNum):
            l = self.architecture[i]
            if l.sparsity:
                l.compileSparsity(self, i, self.options.minibatch_size)

        # Weight decay penalty
        for i in xrange(self.lastArrayNum):
            l = self.architecture[i]
            if l.weightDecay:
                l.compileWeightDecayPenalty(self, i)

        # Error
        XENT = tf.cast(1.0 / self.options.minibatch_size * tf.reduce_sum(tf.pow(tf.sub(self.y, self.varArrayA[-1]),2) * 0.5), tf.float32)
        self.cost = XENT
        #print self.regularize
        #print self.cost
        for err in self.regularize:
            self.cost += err

        # Update output array
        self.outputArray.append(self.cost)
        self.outputArray.append(XENT)
        self.outputArray.append(self.varArrayA[-1])

        # Derivatives
        # All variables to gradArray list to show to Tensorflow on which variables we need an gradient
        gradArray = []
        for i in xrange(self.lastArrayNum):
            for k in self.varWeights[i].keys():
                #print self.varWeights[i][k]
                gradArray.append(self.varWeights[i][k])
                print tf.gradients(self.cost, gradArray)
        self.derivativesArray = tf.gradients(self.cost, gradArray)

        #model = tf.initialize_all_variables()

        #with tf.Session() as sess:
        #    sess.run(model)
        #    print(sess.run(gradArray[3]))
            #print(sess.run(self.regularize[]))
        #print gradArray
        temp = 16
        for i in range(0, len(self.derivativesArray)):
            if self.derivativesArray[i] == None:
                self.derivativesArray[i] = temp*1.0
        print self.derivativesArray
        #RMS
        if self.options.rmsProp:
            for i in xrange(len(self.derivativesArray)):
                #print "RMS Debug"
                #print(gradArray[0].get_shape().as_list())
                mmsp = tf.Variable(np.tile(0.0, gradArray[i].get_shape().as_list()).astype("float32"), name="mmsp%s" % (i + 1))
                self.MMSprev.append(mmsp)
                print mmsp*self.options.rmsProp + (1 - self.options.rmsProp)
                mmsn = self.options.rmsProp * mmsp + (1 - self.options.rmsProp) * tf.pow(self.derivativesArray[i], 2)
                mmsn = tf.clip_by_value(mmsn, self.options.mmsmin, np.finfo(np.float32).max)
                self.MMSnew.append(mmsn)

        #Update Values
        for i in xrange(len(self.derivativesArray)):
            if self.options.rmsProp:
                updateVar = self.options.learnStep * self.derivativesArray[i] / tf.pow(self.MMSnew[i], 0.5)
                self.updatesArray.append((self.MMSprev[i], self.MMSnew[i]))
            else:
                updateVar = self.options.learnStep * self.derivativesArray[i]
            self.updatesArray.append((gradArray[i], gradArray[i] - updateVar))

        sess = tf.InteractiveSession()
        self.train = TensorFlowFunction([self.x, self.y], self.outputArray)

        return self

    def trainCalc(self, X, Y, iteration=10, debug=False, errorCollect=False):  # Need to call trainCompile before
        for i in xrange(iteration):
            error, ent, out = self.train(X, Y)
            self.train_out = out
            if errorCollect:
                self.errorArray.append(ent)
            if debug:
                print ent, error
        return self

    def predictCompile(self):
        #Predict Activation
        for i in xrange(self.lastArrayNum):
            self.architecture[i].compilePredictActivation(self, i)

        sess = tf.InteractiveSession()
        self.predict = TensorFlowFunction([self.x], self.varArrayAc[-1])

        return self

    def predictCalc(self, X, debug=False):  # Need to call predictCompile before
        self.out = self.predict(X)  # Matrix of outputs. Each column is a picture reshaped in vector of features
        if debug:
            print 'out.shape', self.out.get_shape()
        return self

    def paramGetter(self): # Returns the values of model parameters such as [w1, b1, w2, b2] etc.
        model = []
        modell = tf.initialize_all_variables()
        for i in xrange(self.lastArrayNum):  # Possible use len(self.varArrayB) or len(self.varArrayW) instead
            D = dict()
            for k in self.varWeights[i].keys():
                with tf.Session() as session:
                    session.run(modell)
                    D[k] = session.run(self.varWeights[i][k])
                #D[k] = self.varWeights[i][k].get_value()
            model.append(D)
        return model

    def paramSetter(self, loaded):  # Setups loaded model parameters
        assert len(loaded) == self.lastArrayNum, 'Number of loaded and declared layers differs.'
        count = 0
        for l in loaded:
            for k in l.keys():
                self.varWeights[count][k].set_value(np.float32(l[k]))
            count += 1

    def modelSaver(self, folder):  # In cPickle format in txt file
        f = file(folder, "wb")

        #Fix weights with dropout values
        model = self.paramGetter()
        for i in xrange(self.lastArrayNum):
            if self.architecture[i].dropout:
                for k in model[i].keys():
                    if k != 'b':
                        model[i][k] = model[i][k] * self.architecture[i].dropout

        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        self.getStatus()
        return self

    def modelLoader(self, folder):  # Path to model txt file
        f = file(folder, "rb")
        loadedObject = cPickle.load(f)
        f.close()  # Then we need to update W and B parameters

        #Fix model with declared dropout values
        for i in xrange(self.lastArrayNum):
            if self.architecture[i].dropout:
                for k in loadedObject[i].keys():
                    if k != 'b':
                        loadedObject[i][k] = np.true_divide(loadedObject[i][k], self.architecture[i].dropout)

        self.paramSetter(loadedObject)
        self.getStatus()
        return self

    def getStatus(self):  # Its time for troubles
        print self.REPORT
        return self


    def unroll(self):
        l = self.paramGetter()
        meta = []
        count = 0
        for d in l:
            layers_meta = dict()
            for k in sorted(d.keys()):
                layers_meta[k] = d[k].shape
                if count == 0:
                    res = d[k].reshape((-1, ))
                else:
                    res = np.concatenate((res, d[k].reshape((-1, ))))
                count += 1
            meta.append(layers_meta)
        return self


#---------------------------------------------------------------------#
# Options instance
#---------------------------------------------------------------------#


class OptionsStore(object):
    def __init__(self,
                 learnStep=0.01,
                 rmsProp=False,
                 mmsmin=1e-6, # -10
                 rProp=False,
                 minibatch_size=1,
                 CV_size=1):
        self.learnStep = learnStep  # Learning step for gradient descent
        self.rmsProp = rmsProp  # rmsProp on|off
        self.mmsmin = mmsmin  # Min mms value
        self.rProp = rProp  # For full batch only
        self.minibatch_size = minibatch_size
        self.CV_size = CV_size
