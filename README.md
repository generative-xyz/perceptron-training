# perceptron-training

## Usage

```shell
python3 training_user.py -c CONFIG_PATH -d DATASET_PATH -o OUTPUT_PATH
```

- CONFIG_PATH

## Dataset format

The dataset ZIP file should contains folders with name corresponding to the label classes. Each folder should contain images belong to that label class (there is no restriction on image files name). Supported image formats: `.PNG`, `.JPG`/`.JPEG`, `.GIF`, `.BMP`.

For example, suppose that you are about to train a model that can classify three types of animal: dog, cat and mouse. The following is a valid folder structure for the dataset ZIP:


```
.
├── cat/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── dog/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── mouse/
    ├── mouse1.png
    ├── mouse2.jpg
    ├── mouse3.bmp
    └── ...
```

## Config file

Config file format:

```js
{
  "model_name": string,
  "input_dim": [int],
  "structure": [int],
  "activation_name": "ReLU" | "LeakyReLU" | "tanh" | "Sigmoid",
  "val_percent": float,
  "batch_size": int,
  "epoch_num": int,
  "data_augmentation_config": null | {
    "random_flip": "None" | "Horizontal" | "Vertical" | "Both",
    "random_rotation": float,
    "random_zoom": float,
    "random_contrast": float,
    "random_brightness": float
  }
}
```