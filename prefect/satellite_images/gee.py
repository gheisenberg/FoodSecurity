import ee

# Function to get a square around point of interest
# Rural : 5.5km Radius
# Urban : 2 km Radius
def bounding_box(loc, urban_rural, urban_radius, rural_radius):
    """Create bounding box.

    Parameters
    ----------
    loc : Geolocation?
    urban_rural : str
        urban or rural indicator ("u" or "r")
    rural_radius : float
        The radius for buffering rural locations
    urban_radius : float
        The radius for buffering urban locations

    Returns
    -------
    intermediate_box : Polygon?
        returns the square around the location as a Polygon
    """
    if urban_rural is 'U' or urban_rural is 'u':
        size = urban_radius
    else:
        size = rural_radius

    intermediate_buffer = loc.buffer(size)  # buffer radius, half your box width in m
    intermediate_box = intermediate_buffer.bounds()  # Draw a bounding box around the circle
    return (intermediate_box)

def maskClouds(img, MAX_CLOUD_PROBABILITY):
    """Masking of clouds - overlaps the img with the cloud probability map"""
    clouds = ee.Image(img.get('cloud_mask')).select('probability')
    isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY)
    return img.updateMask(isNotCloud)

# Masking of edges
def maskEdges(s2_img):
    return s2_img.updateMask(s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask()))


# In[6]:


def get_image(cluster, survey_name, urban_radius, rural_radius, MAX_CLOUD_PROBABILITY):
    # Get images collections
    s2Sr = ee.ImageCollection('COPERNICUS/S2')
    s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

    # Get time span
    year_uncut = str(cluster["year"])
    year = year_uncut[:year_uncut.rfind('.')]
    if int(year) < 2016:
        START_DATE = ee.Date('2015-06-01')
        END_DATE = ee.Date('2016-07-01')
    else:
        START_DATE = ee.Date(year + '-01-01')
        END_DATE = ee.Date(year + '-12-31')

    # Point of interest (longitude, latidude)
    lat_float = float(cluster["latidude"])
    lon_float = float(cluster["longitude"])
    loc = ee.Geometry.Point([lon_float, lat_float])
    # Region of interest
    region = bounding_box(loc, cluster['urban_rural'], urban_radius, rural_radius)

    # Filter input collections by desired data range and region.
    # criteria = ee.Filter.And(ee.Filter.bounds(region), ee.Filter.date(START_DATE, END_DATE))
    # s2Sr = s2Sr.filter(criteria).map(maskEdges)
    # s2Clouds = s2Clouds.filter(criteria)
    s2Sr = s2Sr.filterBounds(region).filterDate(START_DATE, END_DATE).map(maskEdges)
    s2Clouds = s2Clouds.filterBounds(region).filterDate(START_DATE, END_DATE)

    # Join S2 with cloud probability dataset to add cloud mask.
    s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask').apply(
        primary=s2Sr,
        secondary=s2Clouds,
        condition=ee.Filter.equals(
            leftField='system:index', rightField='system:index')
    )

    maskCloudsWithProb = functools.partial(maskClouds, MAX_CLOUD_PROBABILITY=MAX_CLOUD_PROBABILITY)
    s2CloudMasked = ee.ImageCollection(s2SrWithCloudMask).map(maskCloudsWithProb).median()
    s2CloudMasked = s2CloudMasked.select(
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']).clip(region)
    # Saving location/directory
    # out_dir = os.path.join(survey_dir, cluster["ID-cluster"]+'.tif')
    # geemap.ee_export_image(s2CloudMasked, filename=out_dir, scale=10)
    filename = cluster["ID-cluster"]
    filename = filename.replace(filename[:6], survey_name)
    task = ee.batch.Export.image.toDrive(s2CloudMasked, description=filename, folder='sentinel', scale=10)
    task.start()
    print('Created', filename)
    return loc
