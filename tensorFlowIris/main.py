import tensorflow as tf
import numpy as np
import os
from urllib.request import urlopen

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


class TensorFlowIris(object):
    def __init__(self):
        print("tensor flow Iris")
        self._download_datasets()
        self._load_data_sets()
        # Specify that all features have real-value data
        self.feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
        self.DNNclassifier = tf.contrib.learn.DNNClassifier(feature_columns=self.feature_columns,
                                                            hidden_units=[10, 20, 10],
                                                            n_classes=3,
                                                            model_dir="./tmp/iris_model")
        self.DNNclassifier.fit(input_fn=self.get_train_inputs, steps=2000)
        accuracy_score = self.DNNclassifier.evaluate(input_fn=self.get_test_inputs, steps=1)["accuracy"]
        print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
        predictions = list(self.DNNclassifier.predict(input_fn=self.new_samples))
        print("New Samples, Class Predictions DNN:    {}\n".format(predictions))

        self.kMeansClassifier = tf.contrib.learn.KMeansClustering(num_clusters=3, model_dir="./tmp/k_means_iris")
        self.kMeansClassifier.fit(input_fn=self.get_test_inputs_k_means, steps=2000)
        accuracy_score_k_means = self.kMeansClassifier.evaluate(input_fn=self.get_test_inputs_k_means, steps=1)
        print("\nTest Accuracy: {}\n".format(accuracy_score_k_means))
        predictions_k_means = list(self.kMeansClassifier.predict(input_fn=self.new_samples))
        print("New Samples, Class Predictions: K Means   {}\n".format(predictions_k_means))

    @staticmethod
    def _download_datasets():
        if not os.path.exists(IRIS_TRAINING):
            raw = urlopen(IRIS_TRAINING_URL).read()
            with open(IRIS_TRAINING, "wb") as f:
                f.write(raw)

        if not os.path.exists(IRIS_TEST):
            raw = urlopen(IRIS_TEST_URL).read()
            with open(IRIS_TEST, "wb") as f:
                f.write(raw)

    def _load_data_sets(self):
        self.training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=IRIS_TRAINING,
            target_dtype=np.int,
            features_dtype=np.float32)
        self.test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=IRIS_TEST,
            target_dtype=np.int,
            features_dtype=np.float32)

    # Define the training inputs
    def get_train_inputs(self):
        x = tf.constant(self.training_set.data)
        y = tf.constant(self.training_set.target)
        return x, y

    def get_test_inputs(self):
        x = tf.constant(self.test_set.data)
        y = tf.constant(self.test_set.target)
        return x, y
        # Classify two new flower samples.

    def get_train_inputs_k_means(self):
        x = tf.constant(self.training_set.data)
        y = tf.constant(self.training_set.target)
        return x, None

    def get_test_inputs_k_means(self):
        x = tf.constant(self.test_set.data)
        y = tf.constant(self.test_set.target)
        return x, None

    @staticmethod
    def new_samples():
        return np.array(
            [[6.4, 3.2, 4.5, 1.5],
             [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)


def main():
    TensorFlowIris()

if __name__ == "__main__":
    main()
