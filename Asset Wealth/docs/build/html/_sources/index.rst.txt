.. Asset Wealth Prognosis documentation master file, created by
   sphinx-quickstart on Thu May 26 20:13:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation of Asset wealth forecast for African regions based on remote sensing data
=======================================================================================

Hardware and Software Requirements
-----------------------------------

This code was tested on a system with the following specifications:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* operating system: 20.04.1-Ubuntu SMP
* CPU: AMD EPYC 7443P 24-Core
* GPU: 1x NVIDIA RTX A6000

Software Requirements:
^^^^^^^^^^^^^^^^^^^^^^


* Python Version: 3.8.8
* Tensorflow Version: 2.8
* Keras: 2.8

*Further Python Package requirements are listed in the requirements.txt.*

Data Acquisition & Preprocessing
---------------------------------


#. Calculate Asset Wealth: ``/src/dhs_preparation.py``.
#. Set Parameters for Satellite Data Retrieval inside ``/src/config.py``.
#. Export satellite images from Google Earth Engine:

   #. ``/src/ee_sentinel.py`` for Sentinel-2 Data
   #. ``/src/ee_viirs.py`` for VIIRS Data

#. Move Files to corresponding Preprocessing Folders by using ``/notebooks/split_geotiffs_for_preprocessing.ipynb``.
#. Preprocess GeoTIFFs: ``/src/preprocess_geodata.py``.

Model Training
---------------


#. Set Parameters for Model Training inside ``/src/config.py``.
#. Run ``/src/train_directly.py`` and login to Weights & Bias to track Model Training and Evaluation. 

Notebooks
---------


#. Use ``/notebooks/asset_wealth_analysis.ipynb`` to analyze the calculated Asset Wealth.
#. Use ``/notebooks/asset_wealth_prognosis.ipynb`` to analyze test results and predict 
   Asset Wealth for Mozambique (2016, 2017,2019, 2020 and 2021).

.. toctree::
   :maxdepth: 2
   :caption: See Documentation of Scripts and Notebooks:
   
   src
   notebooks
   