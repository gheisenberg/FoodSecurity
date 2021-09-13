
## Purpose: 
With this code we implemented the CNN for urban regions. The goal is to predict pre-defined waters sources for clusters (see report DHS-surveys) given the Sentinel-2 images of their locations.
This CNN solely predicts water sources for cluster with the residence type urban. We decided to do so as rural and urban regions differs a lot in size and the 3 main water sources are different for rural and urban residence type.

## Requirement: 
The progam is implemented in tensorflow keras. Before running this code, we need the Sentinel-2 images for the clusters (via satellite_images_gee.ipynb and Sorting_sentinel-img_into_zips.ipynb) as well as the water-source file for the labels (via Get_corresponding_GE_HR_survey_files.ipynb and creating_water_source_files.ipynb).  

## Parameters: 
Multiple parameters can be predefined. 

1. Labels (main_labels): The labels which we aim to predict has to be predefined. It is recommended to use the most dominant water sources (to have enough samples,...)

2. Data Set Sizes: The sizes of the training, validation and test set size can also be defined; A 80%/10%/10% (training/validation/testing) would be a typically distribution

3. Cropping/ Input Image Size: Predefining the images size to which we are cropping it. The size of the input images may not be bigger than the size of the smallest satellite image.

4. Clipping values: Decide in which "region" the pixel are allowed to be in. E.g. if the region is 0 to 3000, we would clip an image pixel with value 4302 down to 3000.

5. Channels: Define the channels we want to use and only extract them from our original satellite image; RGB channels would be 4,3,2. (Please note that while using the predefined ResNet50 the images should have exactly 3 input channels)

## Functions: 

### Preprocess Part:
Usually, you only have to run this part once. The functions of this part take a while until finished.

1. get_urban_img: At first, it is necessary to unzip the images from the Sentinel-2 images zip files (each zip file corresponding to one survey) to a temporary folder. Each image which is smaller than 500 in shape is moved to the urban directory (this directory stores all satellite images used as input). 

2. get_labels_df_for_img: a) Removes all images from the urban directory which either do not have a corresponding label or which have a label which is not part of the predefined main labels; b) Creates a label data frame. A row of this data frame contain the name of an image and its corresponding water source label ( in written from (letters) and in categorical form (number)). The data frame contains only the images which have a label which is part of the main labels

3. created_data_sets: Split the urban directory into three directory: training, validation, test. The images are distributed between those directories with a distribution defined before as a parameter.


## Model preporcess/checks
4. calc_mean and calc_std: Calculate the means and the standard deviations channelwise (so a mean/standard deviation for each channel). It is necessary to define the calculation by yourself instead of using the predefined function mean and std as the input to them would be overload the storage/memory.

5. create_class_weights: Returns a dictionary which contains class weights for each label. The class weights should account for a possible imbalance of the labels/classes. The more samples a class has, the lower the weight and vice versa.
The class_weights are added to the model.

6. With training_generator and validation_generator (variables): we only check if the normalization for a single batch worked.

## Neuronal Network

7.train_ds and val_ds: Those are the iterative variable given to the model (basically the input). The function generator basically returns for each (training) step a batch (yield) to train the system. Thanks to yield we do not leave the function as we would with return, but continue to put out the next batch for the next training step and so on.

8. base_model: Currently our base_model is as ResNet50. AS optimizer we use Adam with a learning_rate of 1e-3, the loss is categorical crossentropy.

