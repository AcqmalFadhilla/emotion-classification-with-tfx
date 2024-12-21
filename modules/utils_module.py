import keras_tuner as kt
import tensorflow as tf
import os
import tensorflow_transform as tft
from typing import NamedTuple, Dict, Text, Any
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tfx.components.trainer.fn_args_utils import FnArgs
from keras_tuner.engine import base_tuner
from tfx.components.tuner.component import TunerFnResult


TunerFnResult = NamedTuple('TunerFnResult', [
    ('tuner', base_tuner.BaseTuner),
    ('fit_kwargs', Dict[Text, Any]),
])

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_binary_accuracy',
    mode='max',
    verbose=1,
    patience=10
)

stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
             "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
             "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
             "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
             "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
             "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",
             "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
             "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
             "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
             "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
             "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

LABEL_KEY = "label"
FEATURE_KEY = "text"
VOCAB_SIZE = 1000
SEQ_LENGTH = 100
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32


def transformed_name(key):
  """Renaming transformerd"""
  return key + "_xf"

def preprocessing_fn(inputs):
  """
  Preprocessing inputs features into transformed feature
  Args:
      inputs: map from feature key to raw feature

  Return:
      outputs: map from feature key to transformed feature
  """
  outputs = {}
  feature_key = tf.strings.lower(inputs[FEATURE_KEY])
  feature_key = tf.strings.regex_replace(feature_key, r"(?:<br />)", " ")
  feature_key = tf.strings.regex_replace(feature_key, "n\'t", " not ")
  feature_key = tf.strings.regex_replace(feature_key, r"(?:\'ll |\'re |\'d |\'ve)", " ")
  feature_key = tf.strings.regex_replace(feature_key, r"\W+", " ")
  feature_key = tf.strings.regex_replace(feature_key, r"\d+", " ")
  feature_key = tf.strings.regex_replace(feature_key, r"\b[a-zA-Z]\b", " ")

  outputs[transformed_name(FEATURE_KEY)] = tf.strings.regex_replace(feature_key, r'\b(' + r'|'.join(stopwords) + r')\b\s*', " ")
  outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

  return outputs


def gzip_reader_fn(filenames):
    """Loads compression data

    Args:
        filenames (str): path to the data directory
    Returns:
        TFRecord: compressed data
    """
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, num_epochs, batch_size=64):
    """Generates features and labels for tuning/training
    Args:
        file_pattern: input tfrecord file pattern
        tf_transform_output: A TFTransformOutput
        batch_size: representing the number of consecutive elements of
        returned dataset to combine in a single batch, defaults to 64
    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        num_epochs = num_epochs,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset

def model_builder(hp, vectorize_layer, tuner=False):
  """
    Build machine learning model with optional hyperparameter tuning.

    Args:
        hp (dict or kt.HyperParameters): Hyperparameters for the model.
        vectorize_layer (tf.keras.layers.Layer): Preprocessing layer for text vectorization.
        tuner (bool): Whether to use `kt.HyperParameters` for tuning.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
  # Use given `hp` or fallback to tuner hp
  if tuner:
    hp_embedding = hp.Int('embedding', min_value=16, max_value=128, step=16)
    hp_lstm = hp.Int('lstm_units', min_value=32, max_value=512, step=32)
    hp_dense = hp.Int('dense_units', min_value=32, max_value=512, step=32)
    hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4], default=1e-4)
  else:
      hp_embedding = hp["embedding"]
      hp_lstm = hp["lstm_units"]
      hp_dense = hp["dense_units"]
      hp_lr = hp["learning_rate"]

  # Input layer
  inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)

  # Vectorization and embedding
  x = vectorize_layer(inputs)
  x = Embedding(input_dim=VOCAB_SIZE,
                output_dim=hp_embedding,
                name="embedding")(x)

  # LSTM and Dense layers
  x = Bidirectional(LSTM(hp_lstm))(x)
  x = Dense(hp_dense, activation="relu")(x)

  # Output layer
  outputs = Dense(6, activation="softmax")(x)

  # Compile model
  model = tf.keras.Model(inputs=inputs, outputs=outputs, name="outputs")
  model.compile(
      optimizer=tf.keras.optimizers.Adam(hp_lr),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=["accuracy"]
  )

  model.summary()
  return model

def tuner_fn(fn_args: FnArgs):
    """Tuning the model to get the best hyperparameter based on given args
    Args:
        fn_args: used to train the model as name/value pairs
    Returns:
        TunerFnResult (NamedTuple): object to run model tuner
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = input_fn(fn_args.train_files[0], tf_transform_output, num_epochs=10)
    eval_dataset = input_fn(fn_args.eval_files[0], tf_transform_output, num_epochs=10)

    vectorize_dataset = train_dataset.map(lambda x, y: x[transformed_name(FEATURE_KEY)])
    vectorize_layer = tf.keras.layers.TextVectorization(
      max_tokens=VOCAB_SIZE,
      output_mode="int",
      output_sequence_length=SEQ_LENGTH
    )

    vectorize_layer.adapt(vectorize_dataset)

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: model_builder(hp, vectorize_layer, tuner=True),
        objective=kt.Objective('accuracy', direction='max'),
        max_trials=50,
        seed=42,
        directory=fn_args.working_dir,
        project_name='emotion_kt'
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps,
            'callbacks': [early_stop]
        }
    )

def get_serve_tf_examples_fn(model, tf_transform_output):

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):

        feature_spec = tf_transform_output.raw_feature_spec()

        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        # get predictions using the transformed features
        return model(transformed_features)

    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    # Load the hyperparameters
    hp = fn_args.hyperparameters["values"]

    # Configure callbacks
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )

    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
    # mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir,
    #                                        monitor='val_accuracy',
    #                                        mode='max',
    #                                        verbose=1,
    #                                        save_best_only=False)

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files,
                             tf_transform_output,
                             num_epochs=10,
                             batch_size=TRAIN_BATCH_SIZE)
    val_dataset = input_fn(fn_args.eval_files,
                           tf_transform_output,
                           num_epochs=10,
                           batch_size=EVAL_BATCH_SIZE)

    vectorize_dataset = train_dataset.map(lambda x, y: x[transformed_name(FEATURE_KEY)])
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH
    )
    vectorize_layer.adapt(vectorize_dataset)

    model = model_builder(hp, vectorize_layer, tuner=False)

    model.fit(x=train_dataset,
              validation_data=val_dataset,
              callbacks=[tensorboard_callback, es],
              steps_per_epoch=1000,
              validation_steps=1000,
              epochs=5)

    signatures = {
        'serving_default':
            get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples'))
    }

    #model.save(fn_args.serving_model_dir, save_format='h5', signatures=signatures)
    tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)
