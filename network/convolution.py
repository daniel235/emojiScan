import tensorflow as tf

#planning data

#figure out image size
#grayscale
#training parameters

class faceConvolution:
    def __init__(self):
        self.__init__()
        self.stride = (2,2)
        self.k_size = [5,5]
        self.train_size = None
        self.test_size = None
        self.train = None
        self.test = None

    def prepare_data(self, data):
        # need to get labels
        size = len(data)
        self.train_size = .70 * size
        self.test_size = .30 * size

    # function to create structure of convolution neural network
    def convNet(self, data, inputs):
        input_layer = tf.reshape(data, [-1, 28, 28, 1])
        convOne = tf.layers.conv2d(input_layer, 32, self.k_size, strides=self.stride, activation=tf.nn.relu)
        tf.layers.max_pooling2d(convOne)

    def run(self):
        stride = self.stride
        k_size = self.k_size
        pass