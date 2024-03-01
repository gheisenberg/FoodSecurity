# Start by preprocessing DHS data:
## Create locations file
1. Create locations file:
1.1 Use: create_locations_list.py
1.2 Postprocess in (Q)GIS with gaul layer
1.2.1 use: join attribute by location to get gaul administrative areas into the locations file (use one to many)
1.2.2 repeat this with buffered layers (I do this 2x with max buffer 0.1 degree)
1.2.3 export to csv again
1.3 use: post_processing_gaul_locations.py (cleans up multiples of administrative areas and matches it by country/region/district in the available file, to get the most suitable administrative areas)
## Preprocess DHS data
The DHS data is a wonderful data ressource, but since it is done in multiple countries over multiple versions, they lack a bit of unity. E.g. countries can decide to add custom answer and question items and the answer and question items change over versions.
Therefore a heavy unifying process is necessary: Unify answer and question items
1. use: DHS_unify_answers.ipynb to unify answers in DHS surveys
Note: It is partly based on metadata extraction and automatic processing but mainly works with 
