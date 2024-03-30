import tensorflow as tf
import keras_tuner as kt
import os
import tensorflow_hub as hub
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult

os.environ['TFHUB_CACHE_DIR'] = '/hub_chace'
embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")

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
  """Loads compressed data"""
  return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

def input_fn(file_pattern, tf_transform_output,num_epochs, batch_size=64)->tf.data.Dataset:
  """Get post_transform feature & create batches of data"""

   # Get post_transform feature spec
  transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

   # create bathces of data
  dataset = tf.data.experimental.make_batched_features_dataset(
       file_pattern=file_pattern,
       batch_size=batch_size,
       features=transform_feature_spec,
       reader=gzip_reader_fn,
       num_epochs=num_epochs,
       label_key=transformed_name(LABEL_KEY)
   )
  return dataset

def preprocessing_model_fn(vocab_size, seq_length):
  """
  Prepcoessing inputs feature before entered model
  Args:
      vocab_size: number vocab the converted into token
      seq_length: number of word length
      train_set: data train

  Return:
      encoder: data coverted into token
  """

  encoder = tf.keras.layers.TextVectorization(
      standardize="lower_and_strip_punctuation",
      max_tokens=vocab_size,
      output_mode='int',
      output_sequence_length=seq_length
  )
  return encoder

def model_builder(hp):
  """Build machine learning model"""
  hp_units = hp.Choice('units', [32, 64, 128, 256, 512], default=512)
  hp_lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4], default=1e-4)

  inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
  reshaped_narrative = tf.reshape(inputs, [-1])
  x = vectorize_layer(reshaped_narrative)
  x = tf.keras.layers.Embedding(input_dim=len(vectorize_layer.get_vocabulary()),
                                output_dim=64,
                                mask_zero=True,  #Use masking to handle the variable sequence lengths
                                name="embedding")(x)
  x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hp_units))(x)
  x = tf.keras.layers.Dense(hp_units, activation="relu")(x)
  x = tf.keras.layers.Dense(6, activation="softmax")(x)

  model = tf.keras.Model(inputs=inputs, outputs= x)
  model.compile(optimizer=tf.keras.optimizers.Adam(hp_lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"])
  model.summary()
  return model

def tuner_fn(fn_args: FnArgs)->TunerFnResult:

  tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory=fn_args.working_dir,
                     project_name='kt_hyperband')

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

  train_dataset = input_fn(fn_args.train_files[0],
                            tf_transform_output,
                            10,
                            batch_size=TRAIN_BATCH_SIZE)

  eval_dataset = input_fn(fn_args.eval_files[0],
                           tf_transform_output,
                           10,
                           batch_size=EVAL_BATCH_SIZE)

  stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

  return TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          "callbacks":[stop_early],
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
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

vectorize_layer = preprocessing_model_fn(VOCAB_SIZE, SEQ_LENGTH)
def run_fn(fn_args: FnArgs)-> None:
  log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir, update_freq='batch'
    )
  es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
  mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

  # Load the transform output
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

  train_set = input_fn(fn_args.train_files,
                       tf_transform_output,
                       10,
                       batch_size=TRAIN_BATCH_SIZE)
  val_set = input_fn(fn_args.eval_files,
                     tf_transform_output,
                     10,
                     batch_size=EVAL_BATCH_SIZE)
  vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
                for i in list(train_set)]])

 
  #hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)


  hparams=fn_args.get_best_hyperparameters(num_trials=1)[0]

  #print(f'hyperband parameters: {hparams.get_config()}')

  model = model_builder(hparams)

  model.fit(x = train_set,
            validation_data = val_set,
            callbacks = [tensorboard_callback, es, mc],
            steps_per_epoch = 1000,
            validation_steps= 1000,
            epochs=10)

  signatures = {
        'serving_default':
        get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                                    tf.TensorSpec(
                                    shape=[None],
                                    dtype=tf.string,
                                    name='examples'))
    }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
