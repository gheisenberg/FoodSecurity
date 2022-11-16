"""
Configuration of variables for aquisition of satellite images and model training.
"""

# Data Paths
## Path to Label Data
csv_path = "/mnt/datadisk/data/Projects/asset_wealth/surveys/dhs_data/label_data/"

##########################################################################################################################################
#                                                                                                                                        #
#                                                       Earth Engine Parameters                                                          #
#                                                                                                                                        #
##########################################################################################################################################

gdrive_dir_s2 = '1P4FpvICI0S9vRs8mNvHxzqooK7zPGGnP'
#'https://drive.google.com/drive/folders/1P4FpvICI0S9vRs8mNvHxzqooK7zPGGnP?usp=sharing' #GoogleDrive Directory ID

gdrive_dir_viirs = '' #GoogleDrive Directory ID

download_path_s2 = '/mnt/datadisk/data/Sentinel2/zips'
download_path_viirs = '/mnt/datadisk/data/VIIRS/zips'

# https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.json reduced to country name: ISO CODE
country_code_map = {"Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "American Samoa": "ASM", "Andorra": "AND",
                    "Angola": "AGO", "Anguilla": "AIA", "Antarctica": "ATA", "Antigua and Barbuda": "ATG",
                    "Argentina": "ARG", "Armenia": "ARM", "Aruba": "ABW", "Australia": "AUS", "Austria": "AUT",
                    "Azerbaijan": "AZE", "Bahamas": "BHS", "Bahrain": "BHR", "Bangladesh": "BGD", "Barbados": "BRB",
                    "Belarus": "BLR", "Belgium": "BEL", "Belize": "BLZ", "Benin": "BEN", "Bermuda": "BMU",
                    "Bhutan": "BTN", "Bolivia, Plurinational State of": "BOL", "Bolivia": "BOL",
                    "Bosnia and Herzegovina": "BIH", "Botswana": "BWA", "Bouvet Island": "BVT", "Brazil": "BRA",
                    "British Indian Ocean Territory": "IOT", "Brunei Darussalam": "BRN", "Brunei": "BRN",
                    "Bulgaria": "BGR", "Burkina Faso": "BFA", "Burundi": "BDI", "Cambodia": "KHM", "Cameroon": "CMR",
                    "Canada": "CAN", "Cape Verde": "CPV", "Cayman Islands": "CYM", "Central African Republic": "CAF",
                    "Chad": "TCD", "Chile": "CHL", "China": "CHN", "Christmas Island": "CXR",
                    "Cocos (Keeling) Islands": "CCK", "Colombia": "COL", "Comoros": "COM", "Congo": "COG",
                    "Congo, the Democratic Republic of the": "COD", "Cook Islands": "COK", "Costa Rica": "CRI",
                    "C\u00f4te d'Ivoire": "CIV", "Ivory Coast": "CIV", "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP",
                    "Czech Republic": "CZE", "Denmark": "DNK", "Djibouti": "DJI", "Dominica": "DMA",
                    "Dominican Republic": "DOM", "Ecuador": "ECU", "Egypt": "EGY", "El Salvador": "SLV",
                    "Equatorial Guinea": "GNQ", "Eritrea": "ERI", "Estonia": "EST", "Ethiopia": "ETH",
                    "Falkland Islands (Malvinas)": "FLK", "Faroe Islands": "FRO", "Fiji": "FJI", "Finland": "FIN",
                    "France": "FRA", "French Guiana": "GUF", "French Polynesia": "PYF",
                    "French Southern Territories": "ATF", "Gabon": "GAB", "Gambia": "GMB", "Georgia": "GEO",
                    "Germany": "DEU", "Ghana": "GHA", "Gibraltar": "GIB", "Greece": "GRC", "Greenland": "GRL",
                    "Grenada": "GRD", "Guadeloupe": "GLP", "Guam": "GUM", "Guatemala": "GTM", "Guernsey": "GGY",
                    "Guinea": "GIN", "Guinea-Bissau": "GNB", "Guyana": "GUY", "Haiti": "HTI",
                    "Heard Island and McDonald Islands": "HMD", "Holy See (Vatican City State)": "VAT",
                    "Honduras": "HND", "Hong Kong": "HKG", "Hungary": "HUN", "Iceland": "ISL", "India": "IND",
                    "Indonesia": "IDN", "Iran, Islamic Republic of": "IRN", "Iraq": "IRQ", "Ireland": "IRL",
                    "Isle of Man": "IMN", "Israel": "ISR", "Italy": "ITA", "Jamaica": "JAM", "Japan": "JPN",
                    "Jersey": "JEY", "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN", "Kiribati": "KIR",
                    "Korea, Democratic People's Republic of": "PRK", "Korea, Republic of": "KOR", "South Korea": "KOR",
                    "Kuwait": "KWT", "Kyrgyzstan": "KGZ", "Lao People's Democratic Republic": "LAO", "Latvia": "LVA",
                    "Lebanon": "LBN", "Lesotho": "LSO", "Liberia": "LBR", "Libyan Arab Jamahiriya": "LBY",
                    "Libya": "LBY", "Liechtenstein": "LIE", "Lithuania": "LTU", "Luxembourg": "LUX", "Macao": "MAC",
                    "Macedonia, the former Yugoslav Republic of": "MKD", "Madagascar": "MDG", "Malawi": "MWI",
                    "Malaysia": "MYS", "Maldives": "MDV", "Mali": "MLI", "Malta": "MLT", "Marshall Islands": "MHL",
                    "Martinique": "MTQ", "Mauritania": "MRT", "Mauritius": "MUS", "Mayotte": "MYT", "Mexico": "MEX",
                    "Micronesia, Federated States of": "FSM", "Moldova, Republic of": "MDA", "Monaco": "MCO",
                    "Mongolia": "MNG", "Montenegro": "MNE", "Montserrat": "MSR", "Morocco": "MAR", "Mozambique": "MOZ",
                    "Myanmar": "MMR", "Burma": "MMR", "Namibia": "NAM", "Nauru": "NRU", "Nepal": "NPL",
                    "Netherlands": "NLD", "Netherlands Antilles": "ANT", "New Caledonia": "NCL", "New Zealand": "NZL",
                    "Nicaragua": "NIC", "Niger": "NER", "Nigeria": "NGA", "Niue": "NIU", "Norfolk Island": "NFK",
                    "Northern Mariana Islands": "MNP", "Norway": "NOR", "Oman": "OMN", "Pakistan": "PAK",
                    "Palau": "PLW", "Palestinian Territory, Occupied": "PSE", "Panama": "PAN",
                    "Papua New Guinea": "PNG", "Paraguay": "PRY", "Peru": "PER", "Philippines": "PHL",
                    "Pitcairn": "PCN", "Poland": "POL", "Portugal": "PRT", "Puerto Rico": "PRI", "Qatar": "QAT",
                    "R\u00e9union": "REU", "Romania": "ROU", "Russian Federation": "RUS", "Russia": "RUS",
                    "Rwanda": "RWA", "Saint Helena, Ascension and Tristan da Cunha": "SHN",
                    "Saint Kitts and Nevis": "KNA", "Saint Lucia": "LCA", "Saint Pierre and Miquelon": "SPM",
                    "Saint Vincent and the Grenadines": "VCT", "Saint Vincent & the Grenadines": "VCT",
                    "St. Vincent and the Grenadines": "VCT", "Samoa": "WSM", "San Marino": "SMR",
                    "Sao Tome and Principe": "STP", "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB",
                    "Seychelles": "SYC", "Sierra Leone": "SLE", "Singapore": "SGP", "Slovakia": "SVK",
                    "Slovenia": "SVN", "Solomon Islands": "SLB", "Somalia": "SOM", "South Africa": "ZAF",
                    "South Georgia and the South Sandwich Islands": "SGS", "South Sudan": "SSD", "Spain": "ESP",
                    "Sri Lanka": "LKA", "Sudan": "SDN", "Suriname": "SUR", "Svalbard and Jan Mayen": "SJM",
                    "Swaziland": "SWZ", "Sweden": "SWE", "Switzerland": "CHE", "Syrian Arab Republic": "SYR",
                    "Taiwan, Province of China": "TWN", "Taiwan": "TWN", "Tajikistan": "TJK",
                    "Tanzania, United Republic of": "TZA", "Thailand": "THA", "Timor-Leste": "TLS", "Togo": "TGO",
                    "Tokelau": "TKL", "Tonga": "TON", "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Turkey": "TUR",
                    "Turkmenistan": "TKM", "Turks and Caicos Islands": "TCA", "Tuvalu": "TUV", "Uganda": "UGA",
                    "Ukraine": "UKR", "United Arab Emirates": "ARE", "United Kingdom": "GBR", "United States": "USA",
                    "United States Minor Outlying Islands": "UMI", "Uruguay": "URY", "Uzbekistan": "UZB",
                    "Vanuatu": "VUT", "Venezuela, Bolivarian Republic of": "VEN", "Venezuela": "VEN", "Viet Nam": "VNM",
                    "Vietnam": "VNM", "Virgin Islands, British": "VGB", "Virgin Islands, U.S.": "VIR",
                    "Wallis and Futuna": "WLF", "Western Sahara": "ESH", "Yemen": "YEM", "Zambia": "ZMB",
                    "Zimbabwe": "ZWE"}
##########################################################################################################################################
#                                                                                                                                        #
#                                                       Model Training                                                                   #
#                                                                                                                                        #
##########################################################################################################################################

# GPU
gpu_id = "2"

# Weights and Bias
# wandb_project = #weights and bias project name
# wandb_entity = #weights and bias username
# wandb_dir = "/mnt/datadisk/data/Projects/asset_wealth/run_data/wandb/"

##########################################################################################################################################
#                                                                                                                                        #
#                                                       Model Training                                                                   #
#                                                                                                                                        #
##########################################################################################################################################

# GPU
gpu_id = "2"

# Weights and Bias
# wandb_project = "Asset_Wealth",
# wandb_entity = **SET TO WANDB USERNAME**,
# wandb_dir = **SET DIRECTORY PATH TO STORY WANDB RUNS**

##########################################################################################################################################
#                                                                Dataset                                                                 #
##########################################################################################################################################
# Interval of Min and Max Values for Clipping
clipping_values = [0, 3000]

# Channels to use; [] if all Channels are to be used
channels = []

#  choose whether to use only Urban/Rural Clusters (one of ["u","r"])
urban_rural = "u"

# Image Source to use (one of ["s2", "viirs","s2_viirs"])
img_source = "viirs"

# Number of Channels
if img_source == "viirs":
    channel_size = 3
elif img_source == "s2":
    channel_size = 13
else:
    channel_size = 14

# Pixel Sizes of Images
if urban_rural == "u":
    input_height = 200
    input_width = 200
else:
    input_height = 1000
    input_width = 1000
    
# Countries to use for Training
countries = ["Democratic Republic of Congo", 
             "Ethiopia", 
             "Kenya", 
             "Malawi", 
             "Mozambique", 
             "Rwanda", 
             "Tanzania", 
             "Uganda", 
             "Zambia", 
             "Zimbabwe"]

# Timespan for Training (True for using only data after 2015, else False)
time_restrict = False

# Whether or not to use a subset (for testing)
subset = False

## Paths to Feature Data
feature_data_paths = {
    "s2": {
        "u": {
            "2012_2014": "/mnt/datadisk/data/Sentinel2/preprocessed/asset/urban/2012_2014/",
            "2016_2020": "/mnt/datadisk/data/Sentinel2/preprocessed/asset/urban/2016_2020/",
            "all": "/mnt/datadisk/data/Sentinel2/preprocessed/asset/urban/all/"
        },
        "r": {
            "2012_2014": "/mnt/datadisk/data/Sentinel2/preprocessed/asset/rural/2012_2014/",
            "2016_2020": "/mnt/datadisk/data/Sentinel2/preprocessed/asset/rural/2016_2020/",
            "all": "/mnt/datadisk/data/Sentinel2/preprocessed/asset/rural/all/"
        }
    },
    "viirs": {
        "u": {
            "2012_2014": "/mnt/datadisk/data/VIIRS/preprocessed/asset/urban/2012_2014/",
            "2016_2020": "/mnt/datadisk/data/VIIRS/preprocessed/asset/urban/2016_2020/",
            "all": "/mnt/datadisk/data/VIIRS/preprocessed/asset/urban/all/"
        },
        "r": {
            "2012_2014": "/mnt/datadisk/data/VIIRS/preprocessed/asset/rural/2012_2014/",
            "2016_2020": "/mnt/datadisk/data/VIIRS/preprocessed/asset/rural/2016_2020/",
            "all": "/mnt/datadisk/data/VIIRS/preprocessed/asset/rural/all/"
        }
    },
    "viirs_s2": {
        "u": {
            "2012_2014": "/mnt/datadisk/data/VIIRS_Sentinel2/preprocessed/asset/urban/2012_2014/",
            "2016_2020": "/mnt/datadisk/data/VIIRS_Sentinel2/preprocessed/asset/urban/2016_2020/",
            "all": "/mnt/datadisk/data/VIIRS_Sentinel2/preprocessed/asset/urban/all/"
        },
        "r": {
            "2012_2014": "/mnt/datadisk/data/VIIRS_Sentinel2/preprocessed/asset/rural/2012_2014/",
            "2016_2020": "/mnt/datadisk/data/VIIRS_Sentinel2/preprocessed/asset/rural/2016_2020/",
            "all": "/mnt/datadisk/data/VIIRS_Sentinel2/preprocessed/asset/rural/all/"
        }
    }
}


if time_restrict:
    img_path = feature_data_paths[img_source][urban_rural]["2016_2020"]
    pre2015_path = feature_data_paths[img_source][urban_rural]["2012_2014"]
else:
    img_path = feature_data_paths[img_source][urban_rural]["all"]
    pre2015_path = None
    
##########################################################################################################################################
#                                                       Model Parameters                                                                 #
##########################################################################################################################################
# Number of Epochs
epochs = 20
# Learning rate
lr = 1e-4
# How many pictures are used to train before readjusting weights
batch_size = 16
### The model to use
# available are vgg19, resnet50
model_name = 'vgg19'
# loss function
loss = 'mean_squared_error'
# Min number of epochs to train
early_stopping_patience = 5

# k number of Folds to use for CrossValidation
k = 5
