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
        convOne = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=self.k_size, padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=convOne, pool_size=[2,2], strides=2)

        convTwo = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=self.k_size, padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=convTwo, pool_size=[2,2], strides=2)
        

    def run(self):
        stride = self.stride
        k_size = self.k_size
        pass