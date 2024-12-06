from .config import *
import requests 
import pymysql
import requests
import zipfile
import io
import os
import pandas as pd
import geopandas as gpd
from pyrosm import OSM
from shapely.geometry import box
import numpy as np
from shapely import Polygon
import csv
"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb


# This file accesses the data
Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def housing_upload_join_data(conn, year):
  start_date = str(year) + "-01-01"
  end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print('Selecting data for year: ' + str(year))
  cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
  rows = cur.fetchall()

  csv_file_path = 'output_file.csv'

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print('Storing data for year: ' + str(year))
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  print('Data stored for year: ' + str(year))
  conn.commit()

def download_census_data(code, base_dir=''):
  url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
  extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

  if os.path.exists(extract_dir) and os.listdir(extract_dir):
    print(f"Files already exist at: {extract_dir}.")
    return

  os.makedirs(extract_dir, exist_ok=True)
  response = requests.get(url)
  response.raise_for_status()

  with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(extract_dir)

  print(f"Files extracted to: {extract_dir}")

def load_census_data(code, level='msoa'):
  return pd.read_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-{level}.csv')

def save_data_from_csv(conn, csv_file, table): 
  cur = conn.cursor()
  data_csv = pd.read(csv_file)
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file + f"' INTO TABLE {table} FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  conn.commit()

def join_house_prices_with_oa(connection, oa_boundaries, year): 
    cur = connection.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select * from prices_coordinates_data where date_of_transfer between '{year}-01-01' and '{year}-12-31'")
    prices_coordinates_subset = cur.fetchall()
   
    prices_coordinates_df = pd.DataFrame.from_dict(prices_coordinates_subset)
    prices_coordinates_gdf = gpd.GeoDataFrame(prices_coordinates_df, geometry = gpd.points_from_xy(prices_coordinates_df['longitude'], prices_coordinates_df['latitude']), crs = 'EPSG: 4326')
    prices_coordinates_oa = gpd.sjoin(prices_coordinates_gdf, oa_boundaries, how = 'left', predicate = 'within')
    prices_coordinates_oa = prices_coordinates_oa.drop(columns = ["geometry", "index_right"])
    prices_coordinates_oa = pd.DataFrame(prices_coordinates_oa[~prices_coordinates_oa.isna().any(axis = 1)])
    print(f"Joined data for {year}")
    prices_coordinates_oa.to_csv(f"prices_coordinates_oa_{year}.csv", index = False)
    cur.close()
    connection.close()
    return prices_coordinates_oa


def upload_joined_house_oa(connection, year): 
    csv_file_path = f"./prices_coordinates_oa_{year}.csv"
    cur = connection.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"""
        LOAD DATA LOCAL INFILE '{csv_file_path}' INTO TABLE prices_coordinates_oa_data
        FIELDS TERMINATED BY ',' 
        OPTIONALLY ENCLOSED by '\"' 
        LINES STARTING BY '' 
        TERMINATED BY '\n' 
        IGNORE 1 LINES
        (price, date_of_transfer, postcode, property_type,
        new_build_flag, tenure_type, locality, town_city, district,
        county, primary_addressable_object_name,
        secondary_addressable_object_name, street, country, latitude,
        longitude, db_id, oa_id, lsoa_name);""")
    print(f"Uploaded joined data for {year}")
    connection.commit()
    cur.close()
    connection.close()

def find_houses_bbox(bbox_coords):
   """ Given a bounding box, finds the POIs within the bbox tagged with 'building=residential'
   :param bbox_coords: (west, south, east, north) coordinates 
   :return: GDF with the POIs 
   """
   current_dir = os.path.dirname(__file__)
   data_dir = os.path.join(current_dir, 'data')
   osm_path = os.path.join(data_dir, 'houses.geojson')
   bbox_houses = gpd.read_file(osm_path, bbox = bbox_coords)
   return bbox_houses

def get_bbox_for_region(region_geometry): 
    '''
    Calculates the coordinates of the centroid for the region as well as a bounding box that encompasses the region 
    Args: 
        region_geometry (Polygon): Polygon object that represents the geometry of a certain region 
    Returns: 
        latitude (float): Latitude of centroid
        longitude (float): Longitude of centroid
        box_width (float): Width of bounding box for the region 
        box_height (float): Height of bounding box for the region
    '''
    centroid = region_geometry.centroid
    min_x, min_y, max_x, max_y = region_geometry.bounds
    box_width = max_x - min_x
    box_height = max_y - min_y
    longitude = centroid.x
    latitude = centroid.y
<<<<<<< HEAD
  
=======
    north = latitude + box_width/2
    south = latitude - box_width/2
    east = longitude + box_height/2
    west = longitude - box_height/2
    return (west, south, east, north)

def create_bbox_polygon(row, distance_km): 
    polygon = row['geometry']
    longitude, latitude = row['LONG'], row['LAT']
    box_width = distance_km / 111
    box_height = distance_km / (111 * np.cos(np.radians(latitude)))
    north = latitude + box_width/2
    south = latitude - box_width/2
    east = longitude + box_height/2
    west = longitude - box_height/2
    return Polygon([(west, south), (east, south), (east, north), (west, north)])
>>>>>>> parent of 4719688 (add findhousesbox)
