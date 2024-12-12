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
    min_x, min_y, max_x, max_y = region_geometry.bounds
    return (min_x, min_y, max_x, max_y )

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


def insert_data(connection, table_name, columns, data):
    """
    Inserts list of tuple of row values into any table.
   
    Args:
        connection:  connection object.
        table_name: name of the table
        columns: column names to insert into
        data: list of tuples containing the values

    """
    # Build the query dynamically
    columns_str = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))

    query = f"""
    INSERT INTO {table_name} ({columns_str}) 
    VALUES ({placeholders})
    """

    with connection.cursor() as cursor:
        cursor.executemany(query, data)
    connection.commit()
    connection.close()

def merge_osm_with_naptan_and_oa(osm_df, naptan_df, lsoa_boundaries):
    osm_df = osm_df[osm_df['naptan:AtcoCode'].notna()]
    osm_df = osm_df.merge(naptan_df[['ATCOCode', 'StopType', 'CreationDateTime']], left_on = 'naptan:AtcoCode', right_on = 'ATCOCode')
    osm_df_merged = gpd.sjoin(osm_df, lsoa_boundaries.to_crs(crs = 'epsg:4326')[['LSOA21CD', 'geometry']], predicate = 'within')
    return osm_df_merged

def get_center_coordinates(row):
    """
    Extracts longitude and latitude based on geometry type.

    Args:
        row: A row of the GeoDataFrame.

    Returns:
        A tuple containing longitude and latitude.
    """
    geometry = row['geometry']
    if geometry.type == 'Polygon' or geometry.type == 'MultiPolygon':
        return geometry.centroid.x, geometry.centroid.y
    elif geometry.type == 'Point':
        return geometry.x, geometry.y
    elif geometry.type == 'LineString':
        return geometry.coords[0][0], geometry.coords[0][1]  # Use the first point of the LineString
    else:
        raise Exception

def insert_transport_data(conn, merged_transport_gdf):
    """
    Inserts OSM transport gdfs merged with the NAPTAN, LSOA boundary dataset into transport_node_data table.

    Args:
        row: A row of the GeoDataFrame.

    Returns:
        None
    
    """
    merged_transport_gdf = merged_transport_gdf.drop(columns = 'index_right')
    merged_transport_gdf['longitude'], merged_transport_gdf['latitude'] = zip(*merged_transport_gdf.apply(get_center_coordinates, axis=1))
    merged_transport_gdf['StopType'] = merged_transport_gdf['StopType'].replace(['PLT', 'MET', 'TMU'], 'SUB')
    merged_transport_gdf['StopType'] = merged_transport_gdf['StopType'].replace(['BCT', 'BCS', 'BCQ', 'BST', 'BCE', 'BCP'], 'BUS')
    merged_transport_gdf['StopType'] = merged_transport_gdf['StopType'].replace(['RPLY', 'RLY'], 'RAIL')

    merged_transport_gdf_sql = merged_transport_gdf[['ATCOCode', 'longitude', 'latitude', 'StopType', 'CreationDateTime', 'LSOA21CD']]
    merged_transport_gdf_sql.to_csv('./merged_gdf.csv', index = False)
    # print(railway_station_merged_sql.head())
    import pymysql
    cur = conn.cursor(pymysql.cursors.DictCursor)
    query = """
    LOAD DATA LOCAL INFILE './merged_gdf.csv'
    INTO TABLE transport_node_data
    FIELDS TERMINATED BY ','
    ENCLOSED BY '"'
    LINES TERMINATED BY '\n'
    IGNORE 1 LINES
    (atco_code, longitude, latitude, stop_type, creation_date, lsoa_id);
    """
    cur.execute(query)
    conn.commit()
  
def insert_airport_data(conn, aeroways_merged):
    aeroways_merged['StopType'] = 'AIR'
    aeroways_merged['longitude'], aeroways_merged['latitude'] = zip(*aeroways_merged.apply(get_center_coordinates, axis=1))
    aeroways_merged['ATCOCode'] = [f'AIR_{i}' for i in range(len(aeroways_merged))]
    aeroways_merged_gdf_sql = aeroways_merged[['ATCOCode', 'longitude', 'latitude', 'StopType', 'LSOA21CD']]
    aeroways_merged_gdf_sql.to_csv('./merged_gdf.csv', index = False)
    # print(railway_station_merged_sql.head())
    import pymysql
    cur = conn.cursor(pymysql.cursors.DictCursor)
    query = """
    LOAD DATA LOCAL INFILE './merged_gdf.csv'
    INTO TABLE transport_node_data
    FIELDS TERMINATED BY ','
    ENCLOSED BY '"'
    LINES TERMINATED BY '\n'
    IGNORE 1 LINES
    (atco_code, longitude, latitude, stop_type, lsoa_id);
    """
    cur.execute(query)
    conn.commit()