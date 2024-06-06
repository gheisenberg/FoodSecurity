import tensorflow as tf

# Create a list of file paths
file_paths = ["path/to/image1.tif", "path/to/image2.tif", ...]

# Create a dataset from the file paths
file_paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)


# Define the preprocessing function
def preprocess_image(file_path):
    # Read the image file
    image_string = tf.io.read_file(file_path)
    image = tf.image.decode_tiff(image_string)

    # Select specific channels
    channels_to_keep = [1, 2]  # keep only the green and blue channels
    image = tf.slice(image, [0, 0, 0], [-1, -1, len(channels_to_keep)])

    # Clipping at specified values
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)

    # Interpolation for missing values
    image = tf.image.resize_with_pad(image, target_height=256, target_width=256)

    return image


# Use the map method to apply the preprocessing function to each image
ds = file_paths_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)


# Define the function to convert the image and file path to a TFExample
def to_tfexample(image, file_path):
    image_bytes = tf.io.serialize_tensor(image)
    feature = {'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
               'file_path': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_path.encode()]))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


# Use the map method to apply the to_tfexample function to each image
ds = ds.map(lambda image, file_path: to_tfexample(image, file_path))

# # Use the save method to save the dataset to multiple TFRecord files
# tf.data.experimental.save("path/to/save/TFRecords/", ds, "TFRecord", num_shards=

# import tensorflow as tf

# Create a list of file paths
file_paths = ["path/to/image1.tif", "path/to/image2.tif", ...]

# Create a dataset from the file paths
file_paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)

# Define the preprocessing function


def preprocess_image(file_path):
    # Read the image file
    image_string = tf.io.read_file(file_path)
    image = tf.image.decode_tiff(image_string)

    # Select specific channels
    channels_to_keep = [1, 2]  # keep only the green and blue channels
    image = tf.slice(image, [0, 0, 0], [-1, -1, len(channels_to_keep)])

    # Clipping at specified values
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)

    # Interpolation for missing values
    image = tf.image.resize_with_pad(image, target_height=256, target_width=256)

    return image


# Use the map method to apply the preprocessing function to each image
ds = file_paths_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Create an empty array to store the images
images = tf.TensorArray(tf.float32, size=0)

# Store the images in the array
ds = ds.map(lambda x: images.write(images.size(), x))

# Stack the images to make a single 4D tensor


import tensorflow as tf

# Create a list of file paths
file_paths = ["path/to/image1.tif", "path/to/image2.tif", ...]

# Create a dataset from the file paths
file_paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)


# Define the preprocessing function
def preprocess_image(file_path):
    # Read the image file
    image_string = tf.io.read_file(file_path)
    image = tf.image.decode_tiff(image_string)

    # Select specific channels
    channels_to_keep = [1, 2]  # keep only the green and blue channels
    image = tf.slice(image, [0, 0, 0], [-1, -1, len(channels_to_keep)])

    # Clipping at specified values
    image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=255)

    # Interpolation for missing values
    image = tf.image.resize_with_pad(image, target_height=256, target_width=256)

    return image


# Use the map method to apply the preprocessing function to each image
ds = file_paths_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# # Create an empty array to store the images
# images = tf.TensorArray(tf.float32, size=0



import tensorflow as tf

image_string = tf.io.read_file("path/to/image.tif")
image_decoded = tf.image.decode_image(image_string, channels=2)
image_decoded = tf.image.decode_tiff(image_string)
channels_to_keep = [1,2] # keep only the green and blue channels
image_decoded = tf.slice(image_decoded, [0,0,0], [-1,-1,len(channels_to_keep)])


geotiff_dataset = geotiff_dataset.map(preprocess_image)
geotiff_dataset = geotiff_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)



import tensorflow as tf

def load_geotiff(filepath):
    image_string = tf.io.read_file(filepath)
    image_decoded = tf.image.decode_tiff(image_string)
    return image_decoded

def preprocess_image(image):
    # perform any additional processing here
    return image

# create dataset from the GeoTIFF files in a directory
filepaths = tf.data.Dataset.list_files("path/to/geotiffs/*.tif")
geotiff_dataset = filepaths.map(load_geotiff)

# preprocess the images
geotiff_dataset = geotiff_dataset.map(preprocess_image)


# build the dataset and data input pipeline
# https://pyimagesearch.com/2021/06/14/a-gentle-introduction-to-tf-data-with-tensorflow/
print("[INFO] creating a tf.data input pipeline..")
dataset = tf.data.Dataset.from_tensor_slices(imagePaths)
dataset = (dataset
	.shuffle(1024)
	.map(load_images, num_parallel_calls=AUTOTUNE)
	.cache()
	.repeat()
	.batch(BS)
	.prefetch(AUTOTUNE)
)
# Again, we call the from_tensor_slices function, but this time passing in our imagePaths. Doing so creates a tf.data.Dataset instance where the elements of the dataset are the individual file paths.
#
# We then define the pipeline itself (Lines 45-52)
#
# shuffle: Builds a buffer of 1024 elements from the dataset and shuffles it.
# map: Maps the load_images function across all image paths in the batch. This line of code is responsible for actually loading our input images from disk and parsing the class labels. The AUTOTUNE argument tells TensorFlow to automatically optimize this function call to make it as efficient as possible.
# cache: Caches the result, thereby making subsequent data reads/accesses faster.
# repeat: Repeats the process once we reach the end of the dataset/epoch.
# batch: Returns a batch of data.
# prefetch: Builds batches of data behind the scenes, thereby improving throughput rate.

from tensorflow.keras import layers
#https://archive.is/11M1r#selection-1709.0-2061.2
rescale_layer = tf.keras.Sequential([layers.experimental.preprocessing.Rescaling(1./255)])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(height_factor=(0.2, 0.3), width_factor=(0.2, 0.3)),
    layers.experimental.preprocessing.RandomTranslation(0.3, 0.3, fill_mode='reflect', interpolation='bilinear',)
])

# ###Preprocessing part
# ##W/O writing
# t2 = time.time()
# ds = tf.data.Dataset.list_files(cfg.file_p + '*.tif')
# func = partial(preprocess_file, height=cfg.height, width=cfg.width, channel_l=cfg.channel_l,
#                normalization=geotiff_normalization,
#                replace_nan_value=cfg.replace_nan_value, clipping_values=cfg.clipping_values,
#                std_multiplier=cfg.std_multiplier, means=means, stds=stds)
# #does not calculate yet
# ds = ds.map(lambda x: tf.py_function(func, [x], [tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
# for nr, elem in enumerate(ds):
#     if nr == 0:
#         print('ele', elem)
# #still needs to be written
# print('DS Map PyFunc', time.time() - t2)

##With preprocessing
# t2 = time.time()
# ds = tf.data.Dataset.list_files(outpath + '*.tif')
# #does not calculate yet
# # ds = ds.shuffle(32).map(lambda x: tf.py_function(return_geotiff_as_array, [x], [tf.float32]),
# #             num_parallel_calls=tf.data.AUTOTUNE).cache().batch(16).prefetch(tf.data.AUTOTUNE)
# ds = ds.map(lambda x: tf.py_function(return_geotiff_as_array, [x], [tf.float32]),
#             num_parallel_calls=tf.data.AUTOTUNE)
# for nr, elem2 in enumerate(ds):
#     if nr == 0:
#         print('ele2', elem2)
#     # if nr == 1000:
#     #     break
# #still needs to be written
# print('DS Map with preprocessing PyFunc', time.time() - t2)

##Preprocessing part
# t2 = time.time()
# ds = tf.data.Dataset.list_files(cfg.file_p + '*.tif')
# func = partial(preprocess_file, height=cfg.height, width=cfg.width, channel_l=cfg.channel_l,
#                normalization=geotiff_normalization,
#                replace_nan_value=cfg.replace_nan_value, clipping_values=cfg.clipping_values,
#                std_multiplier=cfg.std_multiplier, means=means, stds=stds, tf_variant=False)
# #does not calculate yet
# ds = ds.map(lambda x: func(x), num_parallel_calls=tf.data.AUTOTUNE)
# for nr, elem in enumerate(ds):
#     if nr == 0:
#         print(elem)
#     break
# #still needs to be written
# print('DS Map', time.time() - t2)
# t2 = time.time()
# ds = tf.data.Dataset.list_files(cfg.file_p + '*.tif')
# # for s in ds:
# #     print(s)
# #     print(s.numpy())
# #     print(str(s.numpy()))
# #     break
# func = partial(preprocess_file, height=cfg.height, width=cfg.width, channel_l=cfg.channel_l,
#                replace_nan_value=cfg.replace_nan_value, clipping_values=cfg.clipping_values,
#                std_multiplier=cfg.std_multiplier, means=means, stds=stds)
# ds.map(lambda x: func([x]))
# except:
#     print('No')

# t1 = time.time()
# print('starting PPE')
# func = partial(preprocess_file, height=cfg.height, width=cfg.width, channel_l=cfg.channel_l,
#                normalization=geotiff_normalization,
#                replace_nan_value=cfg.replace_nan_value, clipping_values=cfg.clipping_values,
#                std_multiplier=cfg.std_multiplier, means=means, stds=stds, write='geotiff', base_p_w=)
# with ProcessPoolExecutor() as executor:
#     results = list(executor.map(func, filenames))
# print('PPE1', time.time() - t1)
# tf.data.Dataset.from_tensor_slices(results)
# print('PPE DS', time.time() - t1)
# arr = np.array(results)
# print(np.mean(arr, axis=(0,2,3)))
# print(np.std(arr, axis=(0,2,3)))


#              , [cfg.height for f in filenames], [cfg.width for f in filenames],
#              [cfg.channel_l for f in filenames], ['Z' for f in filenames], [means for f in filenames],
# [stds for f in filenames], ['local channel mean' for f in filenames], ['outlier' for f in filenames],
# [3 for f in filenames]))

# for file in filenames:
#     preprocess_file(file, height, width, channel_l, normalization='Z', means=means, stds=stds,
#                 replace_nan_value='local channel mean', clipping_values=False, std_multiplier=3)


#
# Yes, you can manually control which layers are active during training and evaluation in tf.keras by using the call method of the layer.
#
# Each layer in tf.keras has a call method that is used to compute its output for a given input. The call method takes an optional training argument that indicates whether the layer is being called during training or evaluation/prediction. By default, the training argument is set to None, but you can pass True or False to the call method to explicitly set the value of training.
#
# For example, if you want to turn off a layer during evaluation/prediction, you can modify the call method of the layer to check the value of the training argument, and return the input unchanged if training is False:
#
# ruby
# Copy code
# class CustomLayer(tf.keras.layers.Layer):
#     def call(self, inputs, training=None):
#         if not training:
#             return inputs
#         # Your layer logic here
#         return ...
# You can use this approach to manually define which layers are active during training and evaluation/prediction, based on your specific needs.
