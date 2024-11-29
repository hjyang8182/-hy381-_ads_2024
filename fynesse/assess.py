from .config import *

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon
import pymysql
import geopandas as gpd
import seaborn as sns
from geopy.distance import geodesic
import numpy as np
"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

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
    min_x, min_y, max_x, max_y = region_geometry.bounds.values[0]
    box_width = max_x - min_x
    box_height = max_y - min_y
    longitude = centroid.x
    latitude = centroid.y
    return (latitude, longitude, box_width, box_height)

def find_poi_area(latitude, longitude, box_width, box_height, tags): 
    '''
    Finds the sum of the areas of POIs near a given pair of coordinates within a bounding box. 
    Args: 
    latitude (float): Latitude of the location 
    longitude (float): Longitude of the location 
    box_width (float): Width of the bounding box surrounding the location
    box_height (float): Height of the bounding box surrounding the location 
    tags (dict): A dictionary of OSM tags representing the POIs
    '''
    north = latitude + box_width/2
    south = latitude - box_width/2
    east = longitude + box_height/2
    west = longitude - box_height/2
    poi_gdf = ox.geometries_from_bbox(north, south, east, west, tags)
    poi_gdf = poi_gdf.loc['relation']
    poi_gdf['area_m2'] = poi_gdf.geometry.to_crs(epsg=3395).area
    area_sum = poi_gdf['area_m2'].sum(axis = 0)
    return area_sum

def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    poi_dict = {}
    box_width = distance_km / 111
    box_height = distance_km / (111 * np.cos(np.radians(latitude)))
    north = latitude + box_width/2
    south = latitude - box_width/2
    east = longitude + box_height/2
    west = longitude - box_height/2
    bbox = (west, south, east, north)
    pois = ox.features_from_bbox(bbox = bbox, tags = tags)
    pois = pd.DataFrame(pois)
    for tag in tags.keys():
      if tag not in pois.columns:
        poi_dict[tag] = 0
      elif type(tags[tag]) is list:
        for item in tags[tag]:
            count = len(pois[pois[tag] == item])
            if np.isnan(count): 
                poi_dict[f"{tag}: {item}"] = 0 
            else: 
                poi_dict[f"{tag}: {item}"] = count
      else:
        poi_dict[tag] = len(pois[pois[tag].notnull()])
    return poi_dict

def count_pois_near_coordinates_box(latitude: float, longitude: float, tags: dict, box_width, box_height) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """
    poi_dict = {}
    north = latitude + box_width/2
    south = latitude - box_width/2
    east = longitude + box_height/2
    west = longitude - box_height/2
    pois = ox.geometries_from_bbox(north, south, east, west, tags)
    pois = pd.DataFrame(pois)
    for tag in tags.keys():
      if tag in pois.columns.values: 
        if type(tags[tag]) is list:
          for item in tags[tag]:
            count = len(pois[pois[tag] == item])
            if np.isnan(count): 
                poi_dict[f"{tag}: {item}"] = 0 
            else: 
                poi_dict[f"{tag}: {item}"] = count
        else:
          count = len(pois[pois[tag].notnull()])
          poi_dict[tag] = count
      else:
        if type(tags[tag]) is list:
          for item in tags[tag]:
            poi_dict[f"{tag}: {item}"] = 0 
        else: 
          poi_dict[tag] = 0
    return poi_dict

def get_poi_gdf(latitude, longitude, tags, distance_km = 1): 
    box_width = distance_km / 111
    box_height = distance_km / (111 * np.cos(np.radians(latitude)))
    north = latitude + box_width/2
    south = latitude - box_width/2
    east = longitude + box_height/2
    west = longitude - box_height/2
    pois = ox.geometries_from_bbox(north, south, east, west, tags)
    pois['area_m2'] = pois.geometry.to_crs(epsg=3395).area
    return pois 

def get_prices_coordinates_from_coords(conn, latitude, longitude, distance_km = 1): 
    cur = conn.cursor(pymysql.cursors.DictCursor)
    box_width = distance_km / 111
    box_height = distance_km / (111 * np.cos(np.radians(latitude)))
    north = latitude + box_width/2
    south = latitude - box_width/2
    east = longitude + box_height/2
    west = longitude - box_height/2
    query = f"SELECT * FROM `prices_coordinates_data` where latitude BETWEEN {south} and {north} and longitude BETWEEN {west} and {east} and date_of_transfer >= '2020-01-01'"
    # query = f'''
    # SELECT *
    # FROM `pp_data` AS pp 
    # INNER JOIN `postcode_data` AS po 
    # ON pp.postcode = po.postcode
    # WHERE latitude BETWEEN {south} and {north} and longitude BETWEEN {west} and {east}'''
    cur.execute(query)
    price_coordinates_data = cur.fetchall()
    return pd.DataFrame(price_coordinates_data)

def plot_buildings_near_coordinates(place_name, latitude: float, longitude: float, distance_km: float = 1.0):
    box_width = distance_km / 111
    box_height = distance_km / (111 * np.cos(np.radians(latitude)))

    north = latitude + box_width/2
    south = latitude - box_width/2
    east = longitude + box_height/2
    west = longitude - box_height/2

    addr_columns = ["addr:housenumber","addr:street", "addr:postcode"]
    pois = get_poi_gdf(latitude, longitude, {"building": True})

    building_addr = pois[pois[addr_columns].notna().all(axis = 1)]
    building_no_addr = pois[pois[addr_columns].isna().any(axis = 1)]
    area = ox.geocode_to_gdf(place_name)
    graph = ox.graph_from_bbox(north, south, east, west)
    nodes, edges = ox.graph_to_gdfs(graph)
    area = ox.geocode_to_gdf(place_name)
    fig, ax = plt.subplots()
    area.plot(ax = ax, facecolor = "white")
    edges.plot(ax = ax, linewidth = 1, edgecolor = "dimgray")
    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    # Plot all POIs
    building_addr.plot(ax=ax, color = "blue", alpha=1 ,markersize=10)
    building_no_addr.plot(ax=ax, color = "red", alpha=1 ,markersize=10) 

def join_prices_coordinates_osm_data(conn, latitude, longitude, distance_km = 1): 
    price_coordinates_data = get_prices_coordinates_from_coords(conn, latitude, longitude, distance_km)
    price_coordinates_data['street'] = price_coordinates_data['street'].str.lower()
    price_coordinates_data['primary_addressable_object_name'] = price_coordinates_data['primary_addressable_object_name'].str.lower()

    pois = get_poi_gdf(latitude, longitude, {"building": True})
    addr_columns = ["addr:housenumber","addr:street", "addr:postcode"]
    
    building_addr = pois[pois[addr_columns].notna().all(axis = 1)]
    building_addr['addr:street'] = building_addr['addr:street'].str.lower()
    building_addr['addr:housenumber'] = building_addr['addr:housenumber'].str.lower()
    building_addr_df = pd.DataFrame(building_addr)
    
    building_no_addr = pois[pois[addr_columns].isna().any(axis = 1)]
    
    merged_on_addr = pd.merge(price_coordinates_data, building_addr_df, left_on = ['street', 'primary_addressable_object_name'], right_on = ['addr:street', 'addr:housenumber'], how = 'inner')
    buildings_not_merged_df = price_coordinates_data[~price_coordinates_data.index.isin(merged_on_addr.index)]
    pois_df = pd.DataFrame(pois)
    buildings_not_merged_df['osmid'] = np.nan
    for index, row in buildings_not_merged_df.iterrows(): 
        db_id = row['db_id']
        longitude, latitude = row['longitude'], row['latitude']
        if pois_df[pois_df['geometry'].apply(lambda x: x.contains(Point(longitude, latitude)))].empty:
            continue
        else:
            buildings_not_merged_df.loc[buildings_not_merged_df['db_id'] == db_id, 'osmid'] = pois_df[pois_df['geometry'].apply(lambda x: x.contains(Point(longitude, latitude)))].index[0][1]
    merged_alt_df = pd.merge(buildings_not_merged_df, pois_df, on = 'osmid')
    full_merged = pd.concat([merged_on_addr, merged_alt_df])
    return full_merged

def find_correlations_with_house_prices(merged_df, latitude, longitude):
    gdf = gpd.GeoDataFrame(merged_df, crs = "ESPG:3395", geometry = merged_df['geometry'])
    city_center = (latitude, longitude)
    gdf['distance_to_center'] = list(map(lambda x: geodesic(x, city_center).kilometers, list(zip(gdf['latitude'], gdf['longitude']))))
    features = ['price', 'area_m2', 'distance_to_center']
    features_df = {feature: gdf[feature].values.tolist() for feature in features}
    features_df = pd.DataFrame(features_df)
    corr_matrix = features_df.corr()
    plt.scatter(features_df['area_m2'].values, features_df['price'].values)
    plt.xlabel("Area (m2)")
    plt.ylabel("Price")
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

