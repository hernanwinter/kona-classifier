import tensorflow as tf
from django.conf import settings
import os
import pandas as pd
import tensorflow_hub as hub
import math

tf.enable_eager_execution()

train_dataset_fp = os.path.join(settings.BASE_DIR, 'classify/movies_metadata.csv')

CSV_COLUMN_NAMES = [
    'original_title', # string
    'vote_average', #int
]

dtypes = {'original_title': 'str', 'vote_average': 'float'}


def load_data(y_name='vote_average'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""

    train_path, test_path = train_dataset_fp, train_dataset_fp

    csv_data = pd.read_csv(train_path, usecols=CSV_COLUMN_NAMES, header=0, dtype=dtypes).sample(frac=1).reset_index(drop=True)

    size = csv_data.size
    train = csv_data[:math.ceil(9*size/10)]
    test = csv_data[math.floor(9*size/10):]

    train_x, train_y = train, train.pop(y_name).map(lambda x: int(round(x)))
    test_x, test_y = test, test.pop(y_name).map(lambda x: int(round(x)))

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def classify(input_text=''):
    # Fetch the data
    (train_x, train_y), (test_x, test_y) = load_data()

    # Feature columns describe how to use the input.
    embedded_text_feature_column = hub.text_embedding_column(
        key="original_title",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=11,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    # Train the Model.
    estimator.train(
        input_fn=lambda: train_input_fn(train_x, train_y, 1000),  #args.batch_size
        steps=5) #args.train_steps

    # Evaluate the model.
    #eval_result = estimator.evaluate(
    #    input_fn=lambda: eval_input_fn(test_x, test_y, 1000))  #args.batch_size

    #print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    predict = {'original_title': [input_text]}
    predictions = estimator.predict(input_fn=lambda: eval_input_fn(predict, labels=None, batch_size=1000))

    class_id = ''
    for pred_dict in predictions:
        template = ('\nPrediction is "{}" ({:.1f}%)')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(class_id, 100 * probability))

    return class_id.item()


#classify('Zoolander')
