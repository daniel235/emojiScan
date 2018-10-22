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
    def convNet(self, features, labels, mode):
        input_layer = tf.reshape(features, [-1, 28, 28, 1])
        convOne = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=self.k_size, padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=convOne, pool_size=[2,2], strides=2)

        convTwo = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=self.k_size, padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=convTwo, pool_size=[2,2], strides=2)

        fcOne = tf.layers.dense(inputs=pool2, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=fcOne)


        #logits layer
        logits = tf.layers.dense(inputs = dropout, units=3)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



    def run(self):
        # creating the estimator
        face_classifier = tf.estimator.Estimator(model_fn=self.convNet, model_dir="./data/testData")
        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook])

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)