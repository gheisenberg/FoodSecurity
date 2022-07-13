# Food Security:
## Asset wealth forecast for African regions based on remote sensing data

### Hardware and Software Requirements
#### This code was tested on a system with the following specifications:
- operating system: 20.04.1-Ubuntu SMP
- CPU: AMD EPYC 7443P 24-Core
- GPU: 1x NVIDIA RTX A6000

#### Software Requirements:
- Python Version: 3.8.8
- Tensorflow Version: 2.8
- Keras: 2.8

*Further Python Package requirements are listed in the requirements.txt.*

### Data Acquisition & Preprocessing

1. Calculate Asset Wealth: ```/src/dhs_preparation.py```.
2. Set Parameters for Satellite Data Retrieval inside ```/src/config.py```.
3. Export satellite images from Google Earth Engine:
   1. ```/src/ee_sentinel.py``` for Sentinel-2 Data
   2. ```/src/ee_viirs.py``` for VIIRS Data
4. Move Files to corresponding Preprocessing Folders by using ```/notebooks/split_geotiffs_for_preprocessing.ipynb```.
5. Preprocess GeoTIFFs: ```/src/preprocess_geodata.py```.

### Model Training

1. Set Parameters for Model Training inside ```/src/config.py```.
2. Run ```/src/train_directly.py``` and login to Weights & Bias to track Model Training and Evaluation. 

### Notebooks

1. Use ```/notebooks/asset_wealth_analysis.ipynb``` to analyze the calculated Asset Wealth.
2. Use ```/notebooks/asset_wealth_prognosis.ipynb``` to analyze test results and predict 
Asset Wealth for Mozambique (2016, 2017,2019, 2020 and 2021).

## [See Project Documentation](code_documentation.pdf)
