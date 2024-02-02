import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import struct
import base64
import json
import tensorflow as tf
import random
import argparse
import sys
import shutil
import traceback
from pathlib import PurePath, Path

tf.keras.utils.disable_interactive_logging()
tf.get_logger().setLevel('ERROR')

EPS = 1e-9
random_seed = 123
shuffle_seed = 123
AUTOTUNE = tf.data.AUTOTUNE

def get_subdirs(directory):
  return [f.path for f in os.scandir(directory) if f.is_dir()]

def get_classes_from_directory(directory):
  class_names = []
  for path in sorted(get_subdirs(directory)):
    while True:
      sub_dirs = get_subdirs(path)
      if len(sub_dirs) == 0:
        break
      path = sub_dirs[0]
    name = PurePath(path).name
    class_names.append(name)
  return class_names

def remove_folder(folder_path):
  try:
    shutil.rmtree(folder_path)
  except OSError:
    pass

def remove_file(file_path):
  try:
    os.remove(file_path)
  except OSError:
    pass

def get_correct_subdirectory(path):
  while True:
    remove_folder(os.path.join(path, '__MACOSX'))
    sub_dirs = get_subdirs(path)
    if len(sub_dirs) == 0:
      sys.exit("Folder '{}' has no subdirectory".format(path))
    elif len(sub_dirs) == 1:
      path = sub_dirs[0]
    else:
      return path
  
def cleanup_dir(input_path):
  for root, dirs, files in os.walk(input_path):
    remove_folder(os.path.join(root, '__MACOSX'))
  for root, dirs, files in os.walk(input_path):
    for file in files:
      path = os.path.join(root, file)
      _, extension = os.path.splitext(path)
      if extension.upper() not in [ ".JPG", ".JPEG", ".PNG", ".GIF", ".BMP" ]:
        remove_file(path)

def read_dataset(input_path, input_dim, val_percent):
  # input_path = get_correct_subdirectory(input_path)
  cleanup_dir(input_path)

  class_names = get_classes_from_directory(input_path)

  ds = tf.keras.utils.image_dataset_from_directory(
    input_path,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=(input_dim[0], input_dim[1]),
    batch_size=None,
    shuffle=True,
    seed=shuffle_seed,
    interpolation='bilinear',
  )

  if val_percent > 0:
    train_ds, val_ds = tf.keras.utils.split_dataset(ds, left_size=1-val_percent/100, shuffle=True, seed=shuffle_seed)
  else:
    train_ds, val_ds = ds, None
  
  return train_ds, val_ds, class_names

def get_data_augmentation(config):
  if config is None:
    return None
  
  data_augmentation = tf.keras.Sequential()
  if config["random_flip"] != "None":
    data_augmentation.add(tf.keras.layers.RandomFlip(get_flip_augmentation(config["random_flip"])))
  if config["random_rotation"] > EPS:
    factor = config["random_rotation"]
    data_augmentation.add(tf.keras.layers.RandomRotation(factor))
  if config["random_zoom"] > EPS:
    factor = config["random_zoom"]
    data_augmentation.add(tf.keras.layers.RandomZoom(height_factor=(-factor, factor), width_factor=(-factor, factor)))
  if config["random_contrast"] > EPS:
    factor = config["random_contrast"]
    data_augmentation.add(tf.keras.layers.RandomContrast(factor))
  if config["random_brightness"] > EPS:
    factor = config["random_brightness"]
    data_augmentation.add(tf.keras.layers.RandomBrightness(factor))

  return data_augmentation

def prepare_dataset(ds, batch_size, data_augmentation=None):
  # Batch all datasets.
  ds = ds.batch(batch_size)

  # Use data augmentation only on the training set.
  if data_augmentation:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)

def leaky_relu(x):
  return tf.keras.activations.relu(x, alpha=0.2)

def get_flip_augmentation(name):
  if name == "Horizontal":
    return "horizontal"
  if name == "Vertical":
    return "vertical"
  if name == "Both":
    return "horizontal_and_vertical"
  sys.exit("Invalid flip augmentation name")

def get_activation_func(name):
  if name == "ReLU":
    return tf.keras.activations.relu
  if name == "tanh":
    return tf.keras.activations.tanh
  if name == "LeakyReLU":
    return leaky_relu
  if name == "Sigmoid":
    return tf.keras.activations.sigmoid
  sys.exit("Invalid activation function name")

def get_model(input_dim, structure, activation_name, class_names):
  activation_func = get_activation_func(activation_name)

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Rescaling(scale=1./127.5, offset=-1))

  # Input layers
  model.add(tf.keras.layers.Flatten(input_shape=(input_dim[0], input_dim[1], 3)))

  # Hidden layers
  for n_node in structure:
    model.add(tf.keras.layers.Dense(n_node, activation=activation_func))
  
  # Output layers
  model.add(tf.keras.layers.Dense(len(class_names)))

  model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

  return model

class CompactedLogCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if "val_loss" in logs:
      print("End epoch {}: loss={:.4f}, acc={:.4f}, val_loss={:.4f}, val_acc={:.4f}"
        .format(epoch+1, logs["loss"], logs["accuracy"], logs["val_loss"], logs["val_accuracy"]))
    else:
      print("End epoch {}: loss={:.4f}, acc={:.4f}"
        .format(epoch+1, logs["loss"], logs["accuracy"]))

def train_model(model, epoch_num, train_ds, val_ds):
  model.fit(train_ds, 
    epochs=epoch_num, 
    validation_data=val_ds,
    callbacks=[CompactedLogCallback()])

def get_traits_for_export(structure, activation_name, epoch_num):
  export_traits = {
    "structure_gen": 'Custom',
    "n_layers": len(structure),
    "max_nodes": max(structure),
    "activation_func": activation_name,
    "epoch_num": epoch_num,
  }

  return export_traits

def compressConfig(data):
  layers = []
  for layer in data["config"]["layers"]:
    if layer["class_name"] == "InputLayer":
      layer_config = {
        "batch_input_shape": layer["config"]["batch_input_shape"]
      }
    elif layer["class_name"] == "Rescaling":
      layer_config = {
        "scale": layer["config"]["scale"],
        "offset": layer["config"]["offset"]
      }
    elif layer["class_name"] == "Dense":
      layer_config = {
        "units": layer["config"]["units"],
        "activation": layer["config"]["activation"]
      }
    else:
      layer_config = None

    res_layer = {
      "class_name": layer["class_name"],
    }
    if layer_config is not None:
      res_layer["config"] = layer_config
    layers.append(res_layer)

  return {
    "config": {
      "layers": layers
    }
  }

def get_model_for_export(model):
  weight_np = model.get_weights()  

  weight_bytes = bytearray()
  for layer in weight_np:
    flatten = layer.reshape(-1).tolist()
    flatten_packed = map(lambda i: struct.pack("@f", i), flatten)
    for i in flatten_packed:
      weight_bytes.extend(i)

  weight_base64 = base64.b64encode(weight_bytes).decode()
  config = json.loads(model.to_json())
  compressed_config = compressConfig(config)
  return weight_base64, compressed_config

def get_file_content(file_path):
  with open(file_path, "r") as f:
    data = f.read()
  return data

def write_to_file(file_path, content):
  f = open(file_path, "w")
  f.write(content)
  f.close()


def get_user_model(data_path, config_path, output_path):
  try:
    print("Loading config...")
    config = json.loads(get_file_content(config_path))

    model_name = config["model_name"]
    input_dim = config["input_dim"]
    structure = config["structure"]
    activation_name = config["activation_name"]
    val_percent = config["val_percent"]
    batch_size = config["batch_size"]
    epoch_num = config["epoch_num"]
    data_augmentation_config = config["data_augmentation_config"]
  except:
    traceback.print_exc()
    sys.exit("Invalid config file")

  try:
    print("Loading dataset...")
    init_train_ds, init_val_ds, class_names = read_dataset(data_path, input_dim, val_percent)
  except:
    traceback.print_exc()
    sys.exit("Error reading dataset")

  try:
    print("Preparing dataset...")
    data_augmentation = get_data_augmentation(data_augmentation_config)
    train_ds = prepare_dataset(init_train_ds, batch_size, data_augmentation)
    val_ds = prepare_dataset(init_val_ds, batch_size) if init_val_ds else None
  except:
    traceback.print_exc()
    sys.exit("Error preparing dataset")


  try:
    print("Building model...")
    tf.random.set_seed(random_seed)
    random.seed(random_seed)
    model = get_model(input_dim, structure, activation_name, class_names)
  except:
    traceback.print_exc()
    sys.exit("Error building model")

  try:
    print("Training model...")
    train_model(model, epoch_num, train_ds, val_ds)
  except:
    traceback.print_exc()
    sys.exit("Error training model")

  try:
    print("Exporting model...")

    export_traits = get_traits_for_export(structure, activation_name, epoch_num)
    weight_base64, compressed_config = get_model_for_export(model)

    inscription = {
      "model_name": model_name,
      "layers_config": compressed_config,
      "weight_b64": weight_base64,
      "training_traits": export_traits,
      "classes_name": class_names
    }
    inscription_json = json.dumps(inscription)

    os.makedirs(name=str(Path(output_path).parent), exist_ok=True)
    write_to_file(output_path, inscription_json)
  except:
    traceback.print_exc()
    sys.exit("Error exporting model")

  print("Finished")


# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("-c", "--config_path", required=True, help="path of config file")
ap.add_argument("-d", "--dataset_path", required=True, help="path of dataset folder")
ap.add_argument("-o", "--output_path", required=True, help="output path of model")

args = vars(ap.parse_args())

data_path = args["dataset_path"]
config_path = args["config_path"]
output_path = args["output_path"]

get_user_model(data_path, config_path, output_path)

