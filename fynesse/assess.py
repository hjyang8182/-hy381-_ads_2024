import pymysql.cursors
from .config import *
from .access import *
import matplotlib.pyplot as plt
import numpy as np
import itertools
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon
import pymysql
import geopandas as gpd
import plotly.express as px
import seaborn as sns
from geopy.distance import geodesic
import ast
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import random
import numpy as np
from scipy.spatial.distance import cdist
import warnings
from matplotlib.patches import Patch
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
    bbox = (west, south, east, north)
    pois = ox.features_from_bbox(bbox, tags)
    pois['area_m2'] = pois.geometry.to_crs(epsg=3395).area
    return pois 

def get_prices_coordinates_from_coords(conn, bbox): 
    cur = conn.cursor(pymysql.cursors.DictCursor)
    west, south, east, north = bbox
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
    pois = get_poi_gdf(latitude, longitude, {'building': True})

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

def plot_corr_matrix(data_df):
    """
        Given a dataframe with multiple features (that can include labels), plot the correlation matrix between the features
    """ 
    feature_df = {feature: data_df[feature].values.tolist() for feature in data_df.columns.values}
    feature_df = pd.DataFrame(feature_df)
    corr_matrix = feature_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)


# TASK 1. PREDICT STUDENT POPULATION   
def find_all_poi_counts(poi_dfs, oa_poi_df, tags): 
    """
        Given a list of POI gdfs with the OA boundary gdf, find the number of POIs that is within an OA boundary
        param: poi_dfs - list of osm node gdfs with index = 'OA21CD'
    """
    poi_counts = []
    for i in range(len(poi_dfs)):
        poi_df = poi_dfs[i]
        tag = tags[i]
        poi_counts.append(find_student_poi_count(poi_df, oa_poi_df, tag).set_index('OA21CD'))
    all_poi_counts = pd.concat(poi_counts, axis  = 1)
    all_poi_counts = all_poi_counts.fillna(0)
    return all_poi_counts

def find_student_poi_count(poi_df, oa_poi_df, tag): 
    joined_poi_oa = gpd.sjoin(oa_poi_df, poi_df, predicate = 'contains')
    oa_counts = joined_poi_oa.groupby('OA21CD').size()
    oa_counts = oa_counts.reset_index()
    oa_counts = oa_counts.rename(columns = {0 : tag})
    return oa_counts


# TASK 2: TRANSPORT FACILITY EFFECT ON HOUSE PRICES
def join_osm_transaction_data(osm_df : pd.DataFrame, transaction_df: pd.DataFrame): 
    """
        Given the transaction data on houses and a gdf consisting of OSMs labelled with tag 
        building = residential, join the two together based on address and whether or not the longitude, latitude coordinate
        of the transaction data falls within the POI
        
        params: 
        - osm_df: gdf consisting of the OSM POI data
        - transaction_df: df consisting of the transaction data from prices_coordinates_oa_data

        returns: 
        - full_merged: gdf consisting of joined OSM POI and transaction data, with log_10 house price values
    """

    transaction_df['street'] = transaction_df['street'].str.lower()
    transaction_df['primary_addressable_object_name'] = transaction_df['primary_addressable_object_name'].str.lower()

    addr_columns = ["addr:housenumber","addr:street"]
    

    building_addr = osm_df[osm_df[addr_columns].notna().all(axis = 1)]
    building_addr['addr:street'] = building_addr['addr:street'].str.lower()
    building_addr['addr:housenumber'] = building_addr['addr:housenumber'].str.lower()
    building_addr_df = pd.DataFrame(building_addr)
    
    merged_on_addr = pd.merge(transaction_df, building_addr_df, left_on = ['street', 'primary_addressable_object_name'], right_on = ['addr:street', 'addr:housenumber'], how = 'inner')

    transactions_not_merged = transaction_df[~transaction_df.index.isin(merged_on_addr.index)]
    transactions_not_merged = gpd.GeoDataFrame(transactions_not_merged, geometry = gpd.points_from_xy(transactions_not_merged['longitude'], transactions_not_merged['latitude']))
    transactions_not_merged = transactions_not_merged.set_crs('epsg:4326')
    osm_df = osm_df.set_crs('epsg:4326')
    merged_on_coord = gpd.sjoin_nearest(transactions_not_merged, osm_df)
    merged_on_coord = merged_on_coord.drop(columns = 'index_right')

    cols = ['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type', 'primary_addressable_object_name', 'secondary_addressable_object_name', 'street', 'latitude', 'longitude', 'db_id', 'geometry']
    merged_on_addr = merged_on_addr[cols]
    merged_on_coord = merged_on_coord[cols]
    full_merged = pd.concat([merged_on_addr, merged_on_coord], ignore_index = True)
    
    full_merged = gpd.GeoDataFrame(full_merged, geometry = 'geometry')
    full_merged['price_log'] = np.log(full_merged['price'])
    return full_merged

def find_transport_bbox(transport_gdf, lad_gdf, transport_type): 
    """
        Given a bounding box, find transport facilities within the box

        params: 
        - transport_gdf: gdf consisting of the OSM transport data
        - lad_gdf: gdf consisting of LAD boundaries data
        - transport_type: classification of transport stops 
            - 'BUS': any transport stops related to buses
            - 'SUB': any transport stops related to underground tube station 
            - 'RAIL': any transport stops related to overground tube stations
            - 'AIR': any transport stops related to airports, aeroways
        returns: 
        - transport_gdf: gdf consisting of queried transport OSM data 
    """
    type_codes = []
    if transport_type == 'BUS': 
        type_codes = ('BCT', 'BCS', 'BCQ', 'BST', 'BCE', 'BCP')
    elif transport_type == 'SUB': 
        type_codes = ('PLT', 'MET', 'TMU')
    elif transport_type == 'RAIL': 
        type_codes = ('RPLY', 'RLY')
    elif transport_type == 'AIR':
        transport_gdf = gpd.sjoin(transport_gdf, lad_gdf, predicate = 'within')
        transport_gdf = transport_gdf.drop(columns = 'index_right')
        return transport_gdf
    else: 
       pass
    # Transport node data seems to be missing a lot of values
    # cur = conn.cursor(pymysql.cursors.DictCursor)
    # cur.execute(f"select * from transport_node_data where longitude between {south} and {north} and latitude between {east} and {west} and stop_type in {type_codes}")
    transport_gdf = gpd.sjoin(transport_gdf, lad_gdf, predicate = 'within')
    transport_gdf = transport_gdf.drop(columns = 'index_right')
    transport_gdf = transport_gdf[np.isin(transport_gdf['StopType'], type_codes)]
    transport_gdf = gpd.GeoDataFrame(transport_gdf)
    return transport_gdf

def find_transport_bbox_sql(conn, bbox, transport_type): 
    cur = conn.cursor(pymysql.cursors.DictCursor)
    west, south, east, north = bbox
    cur.execute(f"select * from transport_node_data where longitude between {west} and {east} and latitude between {south} and {north} and stop_type = '{transport_type}'")
    transport_gdf = gpd.GeoDataFrame(cur.fetchall())
    return transport_gdf

def find_transaction_lad_id(conn, lad_id, lad_boundaries): 
    lad_row = lad_boundaries[lad_boundaries['LAD21CD'] == lad_id]
    lad_bbox = lad_row.bbox.values[0]
    return find_transaction_bbox_after_2020(conn, lad_bbox)

def find_transport_lad_id(transport_gdf, transport_type, lad_id, lad_boundaries): 
    lad_row = lad_boundaries[lad_boundaries['LAD21CD'] == lad_id]
    lad_gdf = gpd.GeoDataFrame({'geometry': lad_row.geometry})
    return find_transport_bbox(transport_gdf, lad_gdf, transport_type)

def find_transport_lad_id_sql(conn, lad_id, lad_boundaries, transport_type): 
    lad_row = lad_boundaries[lad_boundaries['LAD21CD'] == lad_id]
    lad_bbox = lad_row.bbox.values[0]
    return find_transport_bbox_sql(conn, lad_bbox, transport_type)

def find_transaction_bbox_after_2020(conn, bbox): 
    """
        Given a bounding box, find house purchase transactions within the box

        params: 
        - bbox: (west, south, east, north) coordinates of the bounding box
        returns: 
        - transaction_df: df consisting of queried transaction data from prices_coordinates_oa_data
    """
    west, south, east, north = bbox
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select * from prices_coordinates_oa_data where latitude between {south} and {north} and longitude between {west} and {east} and date_of_transfer >= 2020-01-01")
    transaction_df = pd.DataFrame(cur.fetchall())
    return transaction_df

def find_all_transactions_bbox(conn, bbox): 
    west, south, east, north = bbox
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select * from prices_coordinates_oa_data where latitude between {south} and {north} and longitude between {west} and {east} and date_of_transfer >= 2020-01-01")
    transaction_df = pd.DataFrame(cur.fetchall())
    return transaction_df

def find_residential_buildings(conn, lad_id, building_dfs): 
    """
        Given a LAD id and all building=residential OSM data, find POIs within the LAD

        params: 
        - lad_id: unique iD of the LAD
        - building_dfs: gdf consisting of all building=residential OSM pois

        returns: 
        - buildings_gdf: gdf consisting of queried building=residential POIs from within the LAD
    """
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select unique lsoa_id from oa_translation_data where lad_id = '{lad_id}'")
    lsoa_ids = list(map(lambda x : x['lsoa_id'], cur.fetchall()))
    buildings = []
    for i in range(len(building_dfs)): 
        building_df = building_dfs[i]
        buildings.append(building_df[np.isin(building_df['lsoa_id'].values, lsoa_ids)])
    buildings_df = pd.concat(buildings)
    buildings_gdf = gpd.GeoDataFrame(buildings_df, geometry = 'geometry')
    buildings_gdf = buildings_gdf.drop_duplicates('index_right')
    buildings_gdf = buildings_gdf.drop(columns = 'index_right')
    return buildings_gdf

def find_joined_osm_transaction_data_lsoa(conn, lsoa_id, building_dfs): 
    buildings = []
    for i in range(len(building_dfs)): 
        building_df = building_dfs[i]
        buildings.append(building_df[building_df['lsoa_id'] == lsoa_id])
    buildings_df = pd.concat(buildings)
    buildings_gdf = gpd.GeoDataFrame(buildings_df, geometry = 'geometry')
    buildings_gdf = buildings_gdf.drop_duplicates('index_right')
    buildings_gdf = buildings_gdf.drop(columns = 'index_right')
    transactions_lsoa = find_transaction_lsoa(conn, lsoa_id)
    if transactions_lsoa.empty: 
        return
    joined_data = join_osm_transaction_data(buildings_gdf, transactions_lsoa)
    if joined_data.empty: 
        return gpd.GeoDataFrame(transactions_lsoa, geometry = gpd.points_from_xy(transactions_lsoa['longitude'], transactions_lsoa['latitude']))
    return joined_data

def plot_lad_prices(conn, lad_id, building_dfs, lad_boundaries, transport_gdf, transport_type):
    """
        transport type: 
            'BUS' :  bus stops, station entrances
            'SUB' : tube, tram, underground platform entrances
            'RAIL' : railway stations 
            'AIR' : airport, air access areas
    """ 
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        buildings_gdf = find_residential_buildings(conn, lad_id, building_dfs)
        lad_row = lad_boundaries[lad_boundaries['LAD21CD'] == lad_id]
        lad_name = lad_row.LAD21NM.values[0]
        lad_bbox = lad_row.bbox.values[0]
        lad_gdf = gpd.GeoDataFrame({'geometry': lad_row.geometry})

        transport_gdf = find_transport_bbox(transport_gdf, lad_gdf, transport_type)
        house_transactions = find_transaction_bbox_after_2020(conn, lad_bbox)
        osm_prices_merged = join_osm_transaction_data(buildings_gdf, house_transactions)

        osm_prices_merged = gpd.sjoin(osm_prices_merged, lad_gdf, predicate = 'within')
        fig, ax = plt.subplots()
        lad_gdf.plot(ax = ax, facecolor = 'white', edgecolor = 'dimgray')
        osm_prices_merged = osm_prices_merged.sort_values(by='price', ascending=True)
        osm_prices_merged.plot(column = 'price_log', ax = ax, legend = True, cmap = 'viridis')
        transport_gdf.plot(ax = ax, color = 'red', markersize = 10)

        # custom_patch = mpatches.Patch(color='red', label='Transport Facilities')
        # ax.legend(handles=[custom_patch], title="Legend")
        plt.title(f"log Price of Houses in {lad_name}")
        plt.show()

def plot_lad_prices_random_subset(conn, lad_ids, building_dfs, lad_boundaries, transport_gdf, transport_type):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        fig, ax = plt.subplots(3, 3, figsize=(12,12))
        for i in range(3):
            for j in range(3): 
                lad_id = lad_ids[i * j]
                buildings_gdf = find_residential_buildings(conn, lad_id, building_dfs)
                lad_row = lad_boundaries[lad_boundaries['LAD21CD'] == lad_id]
                lad_name = lad_row.LAD21NM.values[0]
                lad_geom = lad_row.geometry.values[0]
                lad_bbox = lad_row.bbox.values[0]
                lad_gdf = gpd.GeoDataFrame({'geometry': lad_row.geometry})

                transport_gdf = find_transport_bbox(transport_gdf, lad_gdf, transport_type)
                house_transactions = find_transaction_bbox_after_2020(conn, lad_bbox)
                osm_prices_merged = join_osm_transaction_data(buildings_gdf, house_transactions)

                try: 
                    osm_prices_merged = gpd.sjoin(osm_prices_merged, lad_gdf, predicate = 'within')
                    lad_gdf.plot(ax = ax[i,j], facecolor = 'white', edgecolor = 'dimgray')
                    osm_prices_merged.plot(column = 'price_log', ax = ax[i,j], legend = True, cmap = 'viridis')
                    transport_gdf.plot(ax = ax[i,j], color = 'red', markersize = 10)
                    ax[i, j].set_title(f"log Price of Houses in {lad_name}", fontsize=8, fontweight='light')
                except: 
                    pass
                # custom_patch = mpatches.Patch(color='red', label='Transport Facilities')
                # ax.legend(handles=[custom_patch], title="Legend")
                plt.tight_layout()
                plt.show()

def find_avg_lsoa_price_in_lad(conn, lad_id, lad_boundaries): 
    lad_row = lad_boundaries[lad_boundaries['LAD21CD'] == lad_id]
    lad_name = lad_row.LAD21NM.values[0]
    lad_geom = lad_row.geometry.values[0]
    lad_bbox = lad_row.bbox.values[0]

    cur = conn.cursor(pymysql.cursors.DictCursor)
    west, south, east, north = lad_bbox
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select lsoa_id, avg(price) as avg_price from prices_coordinates_oa_data where date_of_transfer >= '2020-01-01' and latitude BETWEEN {south} and {north} and longitude BETWEEN {west} and {east}  group by lsoa_id")
    avg_lsoas_col = cur.fetchall()
    return avg_lsoas_col 

def plot_avg_lsoa_prices_in_lad(conn, lad_id, lad_boundaries, lsoa_boundaries, transport_gdf, transport_type):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        avg_lsoa_prices = find_avg_lsoa_price_in_lad(conn, lad_id, lad_boundaries)
        avg_lsoa_prices_df = pd.DataFrame(avg_lsoa_prices)
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute(f"select lsoa_id from oa_translation_data where lad_id = '{lad_id}'")
        lad_lsoa_ids = list(map(lambda x : x['lsoa_id'], cur.fetchall()))
        lsoa_boundaries = lsoa_boundaries[np.isin(lsoa_boundaries['LSOA21CD'], lad_lsoa_ids)]
        lsoa_avg_merged = avg_lsoa_prices_df.merge(lsoa_boundaries[['LSOA21CD', 'geometry']], left_on = 'lsoa_id', right_on = 'LSOA21CD')
        lsoa_avg_merged_gdf = gpd.GeoDataFrame(lsoa_avg_merged, geometry = 'geometry')
        lad_row = lad_boundaries[lad_boundaries['LAD21CD'] == lad_id]
        lad_gdf = gpd.GeoDataFrame({'geometry': lad_row.geometry})

        lsoa_avg_merged_gdf = gpd.sjoin(lsoa_avg_merged_gdf, lad_gdf, predicate = 'within')
        transport_gdf = find_transport_bbox(transport_gdf, lad_gdf, transport_type)
        fig, ax = plt.subplots()
        lad_gdf.plot(ax = ax, facecolor = 'white', edgecolor = 'dimgray')
        lsoa_avg_merged_gdf['avg_price'] = np.log(lsoa_avg_merged_gdf['avg_price'].astype(float))
        lsoa_avg_merged_gdf.plot(ax = ax, column = 'avg_price', cmap = 'viridis', legend=True)
        transport_gdf.plot(ax = ax, color = 'red')
        plt.title("Average House Price per LSOA in LAD")
        custom_patch = mpatches.Patch(color='red', label='Transport Facility')
        plt.legend(handles=[custom_patch], title="Legend", loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.5)
    
def find_distance_to_closest_transport(connection, lsoa_id, transport_lad):
    cur = connection.cursor(pymysql.cursors.DictCursor)
    houses_lsoa = find_transaction_lsoa(connection, lsoa_id)
    if houses_lsoa.empty: 
        return
    houses_lsoa = gpd.GeoDataFrame(houses_lsoa, geometry = gpd.points_from_xy(houses_lsoa['longitude'], houses_lsoa['latitude']))
    distance_df = compute_pairwise_distances(houses_lsoa, transport_lad)
    closest_distances_df = find_closest_points(distance_df)
    closest_distances_with_prices = closest_distances_df.merge(houses_lsoa, left_on = 'house_index', right_index = True)
    transport_lad = transport_lad.reset_index(drop = True)
    closest_distances_with_prices_transport = closest_distances_with_prices.merge(transport_lad[['creation_date', 'atco_code', 'stop_type']], left_on = 'transport_index', right_index = True)
    return closest_distances_with_prices_transport

def find_median_pct_inc_after_transport_vs_dist(conn, lad_id, transport_type, lad_boundaries, num_lsoas):
    pct_incs = []
    avg_dists = []
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"SELECT unique lsoa_id FROM oa_translation_data where lad_id = '{lad_id}' ORDER BY RAND() LIMIT {num_lsoas}")
    lsoa_ids = list(map(lambda x : x['lsoa_id'], cur.fetchall()))
    transport_lad = find_transport_lad_id_sql(conn, lad_id, transport_type)
    if transport_lad.empty: 
        return
    for lsoa_id in lsoa_ids:
        distance_df = find_distance_to_closest_transport(conn, lsoa_id, transport_lad)
        if distance_df is None: 
            continue
        distance_df_grouped = distance_df.groupby(['lsoa_id', 'transport_index'])
        for idx, distance_df in distance_df_grouped: 
            creation_year = pd.to_datetime(distance_df['CreationDateTime']).dt.year
            median_before = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) < creation_year]['price'].values)
            median_after = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) >= creation_year]['price'].values)
            median_before = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) < creation_year]['price'].values)
            median_after = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) >= creation_year]['price'].values)
            pct_inc = (median_after - median_before)/median_before * 100
            avg_dist = np.mean(distance_df['distance'].values)
            pct_incs.append(pct_inc)
            avg_dists.append(avg_dist)
    return pct_incs, avg_dists

def find_median_pct_inc_after_transport_vs_car_ava(conn, lad_id, transport_gdf, transport_type, lad_boundaries, num_lsoas = 5):
    pct_incs = []
    no_car_proportions = []
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"SELECT unique lsoa_id FROM oa_translation_data where lad_id = '{lad_id}' ORDER BY RAND() LIMIT {num_lsoas}")
    lsoa_ids = list(map(lambda x : x['lsoa_id'], cur.fetchall()))
    transport_lad = find_transport_lad_id(transport_gdf, transport_type, lad_id, lad_boundaries)
    if transport_lad.empty: 
        return
    for lsoa_id in lsoa_ids:
        # nearest transport facility to the lsoa
        distance_df = find_distance_to_closest_transport(conn, lsoa_id, transport_lad)
        if distance_df is None: 
            continue
        distance_df_grouped = distance_df.groupby(['lsoa_id', 'transport_index'])
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute(f"select * from car_availability_data where lsoa_id = '{lsoa_id}'")
        car_availability = cur.fetchall()[0]['no_car_proportion']
        for idx, distance_df in distance_df_grouped: 
            creation_year = pd.to_datetime(distance_df['CreationDateTime']).dt.year
            median_before = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) < creation_year]['price'].values)
            median_after = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) >= creation_year]['price'].values)
            median_before = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) < creation_year]['price'].values)
            median_after = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) >= creation_year]['price'].values)
            pct_inc = (median_after - median_before)/median_before * 100
            no_car_proportions.append(car_availability)
            pct_incs.append(pct_inc)
    return pct_incs, no_car_proportions

def find_median_pct_inc_after_transport_vs_travel_method(conn, lad_id, transport_gdf, transport_type, lad_boundaries, num_lsoas = 5):
    pct_incs = []
    transport_usages = []
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"SELECT unique lsoa_id FROM oa_translation_data where lad_id = '{lad_id}' ORDER BY RAND() LIMIT {num_lsoas}")
    lsoa_ids = list(map(lambda x : x['lsoa_id'], cur.fetchall()))
    transport_lad = find_transport_lad_id(transport_gdf, transport_type, lad_id, lad_boundaries)
    if transport_lad.empty: 
        return
    for lsoa_id in lsoa_ids:
        # nearest transport facility to the lsoa
        distance_df = find_distance_to_closest_transport(conn, lsoa_id, transport_lad)
        if distance_df is None: 
            continue
        distance_df_grouped = distance_df.groupby(['lsoa_id', 'transport_index'])
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute(f"select * from method_of_travel_data where lsoa_id = '{lsoa_id}'")
        transport_usage  = cur.fetchall()[0][transport_type]
        for idx, distance_df in distance_df_grouped: 
            creation_year = pd.to_datetime(distance_df['CreationDateTime']).dt.year
            median_before = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) < creation_year]['price'].values)
            median_after = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) >= creation_year]['price'].values)
            median_before = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) < creation_year]['price'].values)
            median_after = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) >= creation_year]['price'].values)
            pct_inc = (median_after - median_before)/median_before * 100
            transport_usages.append(transport_usage)
            pct_incs.append(pct_inc)
    return pct_incs, transport_usages

def find_median_pct_inc_after_transport_vs_house_type(conn, lad_id, transport_gdf, transport_type, lad_boundaries, num_lsoas = 5):
    combined_df = pd.DataFrame(columns = ['D', 'S', 'T', 'F', 'O', 'Y', 'N', 'pct_change'])
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"SELECT unique lsoa_id FROM oa_translation_data where lad_id = '{lad_id}' ORDER BY RAND() LIMIT {num_lsoas}")
    lsoa_ids = list(map(lambda x : x['lsoa_id'], cur.fetchall()))
    transport_lad = find_transport_lad_id(transport_gdf, transport_type, lad_id, lad_boundaries)
    if transport_lad.empty: 
        return
    for lsoa_id in lsoa_ids:
        # nearest transport facility to the lsoa
        distance_df = find_distance_to_closest_transport(conn, lsoa_id, transport_lad)
        if distance_df is None: 
            continue
        distance_df_grouped = distance_df.groupby(['lsoa_id', 'transport_index'])
        cur = conn.cursor(pymysql.cursors.DictCursor)
        cur.execute(f"select * from car_availability_data where lsoa_id = '{lsoa_id}'")
        for idx, distance_df in distance_df_grouped: 
            creation_year = pd.to_datetime(distance_df['CreationDateTime']).dt.year
            median_before = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) < creation_year]['price'].values)
            median_after = np.median(distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year) >= creation_year]['price'].values)
            pct_inc = (median_after - median_before)/median_before * 100
            if np.isnan(pct_inc):
                continue
            property_type_counts = distance_df['property_type'].value_counts().to_dict()
            new_build_counts = distance_df['new_build_flag'].value_counts().to_dict()
            merged_counts = property_type_counts | new_build_counts
            merged_counts['pct_change'] = pct_inc
            combined_df = pd.concat([combined_df, pd.DataFrame([merged_counts]).fillna(0)], ignore_index=True)
    combined_df[['D', 'S', 'T', 'F', 'O']] = combined_df[['D', 'S', 'T', 'F', 'O']].div(combined_df[['D', 'S', 'T', 'F', 'O']].sum())
    combined_df[['Y', 'N']] = combined_df[['Y','N']].div(combined_df[['Y','N']].sum())
    combined_df = combined_df.fillna(0)
    return combined_df

def find_yearly_pct_inc_after_transport(conn, lad_id, transport_gdf, transport_type, lad_boundaries, num_lsoas = 5):
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"SELECT unique lsoa_id FROM oa_translation_data where lad_id = '{lad_id}' ORDER BY RAND() LIMIT {num_lsoas}")
    lsoa_ids = list(map(lambda x : x['lsoa_id'], cur.fetchall()))
    transport_lad = find_transport_lad_id(transport_gdf, transport_type, lad_id, lad_boundaries)
    if transport_lad.empty: 
        return
    years_after_creation_vals = np.array([])
    pct_change_vals = np.array([])
    for lsoa_id in lsoa_ids:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            distance_df = find_distance_to_closest_transport(conn, lsoa_id, transport_lad)
            if distance_df is None: 
                continue
            distance_df_grouped = distance_df.groupby(['lsoa_id', 'transport_index'])
            for idx, distance_df in distance_df_grouped: 
                creation_year = pd.to_datetime(distance_df['CreationDateTime']).dt.year
                distance_df_after = distance_df[pd.to_datetime(distance_df['date_of_transfer']).dt.year >= creation_year]
                distance_df_after['years_after_creation'] = pd.to_datetime(distance_df['date_of_transfer']).dt.year  - creation_year
                
                median_prices = distance_df_after.groupby('years_after_creation')['price'].median().reset_index()
                median_prices['pct_change'] = median_prices['price'].pct_change()
                median_prices = median_prices.dropna()
                years_after_creation_vals = np.concatenate([years_after_creation_vals, median_prices['years_after_creation'].values])
                pct_change_vals = np.concatenate([pct_change_vals, median_prices['pct_change'].values])
    return years_after_creation_vals, pct_change_vals     

def find_all_features_modified(conn, lad_id, transport_gdf, transport_type, lad_boundaries, num_lsoas = 5):
    feature_df = {}
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"SELECT unique lsoa_id FROM oa_translation_data where lad_id = '{lad_id}' ORDER BY RAND() LIMIT {num_lsoas}")
    lsoa_ids = list(map(lambda x : x['lsoa_id'], cur.fetchall()))
    transport_lad = find_transport_lad_id_sql(conn, lad_id, lad_boundaries, transport_type)
    if transport_lad.empty: 
        return
    transport_lad = gpd.GeoDataFrame(transport_lad, geometry = gpd.points_from_xy(transport_lad['longitude'], transport_lad['latitude']))
    pct_incs = []
    transport_usage_vals = []
    car_availability_vals = []
    avg_dists =  []
    for lsoa_id in lsoa_ids:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            distance_df = find_distance_to_closest_transport(conn, lsoa_id, transport_lad)
            if distance_df is None: 
                continue
            distance_df_grouped = distance_df.groupby(['lsoa_id', 'transport_index'])
            cur = conn.cursor(pymysql.cursors.DictCursor)
            cur.execute(f"select mt.*, ca.* from method_of_travel_data as mt join car_availability_data as ca on ca.lsoa_id = mt.lsoa_id where ca.lsoa_id = '{lsoa_id}'")
            census_results  = cur.fetchall()[0]
            transport_usage = census_results[transport_type]
            car_availability = census_results['no_car_proportion']
            for idx, distance_df in distance_df_grouped: 
                creation_year = pd.to_datetime(distance_df['creation_date']).dt.year
                distance_df_after = distance_df[pd.to_datetime(distance_df['date_of_transfer']).dt.year >= creation_year]
                distance_df_after['years_after_creation'] = pd.to_datetime(distance_df['date_of_transfer']).dt.year  - creation_year
                avg_dist = np.mean(distance_df['distance'].values)
                median_before = np.median(
                    distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year < creation_year) & distance_df['price'].notna()]['price'].values
                )
                median_after = np.median(
                    distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year >= creation_year) & distance_df['price'].notna()]['price'].values
                )
                pct_inc = (median_after - median_before)/median_before * 100
                pct_incs.append(pct_inc)
                transport_usage_vals.append(transport_usage)
                car_availability_vals.append(car_availability)
                avg_dists.append(avg_dist)
    features_df = pd.DataFrame({
        'pct_inc': pct_incs,
        'transport_usage': transport_usage_vals, 
        'car_availability': car_availability_vals,
        'avg_dist': avg_dists
        })
    return features_df

def find_all_features_with_house_types(conn, lad_id, transport_type, lad_boundaries, num_lsoas = 5):
    feature_df = {}
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"SELECT unique lsoa_id FROM oa_translation_data where lad_id = '{lad_id}' ORDER BY RAND() LIMIT {num_lsoas}")
    lsoa_ids = list(map(lambda x : x['lsoa_id'], cur.fetchall()))
    transport_lad = find_transport_lad_id_sql(conn, lad_id, lad_boundaries, transport_type)
    if transport_lad.empty: 
        return
    transport_lad = gpd.GeoDataFrame(transport_lad, geometry = gpd.points_from_xy(transport_lad['longitude'], transport_lad['latitude']))
    pct_incs = []
    transport_usage_vals = []
    car_availability_vals = []
    avg_dists =  []

    D_vals = []  # For 'D' column
    S_vals = []  # For 'S' column
    T_vals = []  # For 'T' column
    F_vals = []  # For 'F' column
    O_vals = []  # For 'O' column
    Y_vals = []  # For 'Y' column
    N_vals = []  # For 'N' column

    for lsoa_id in lsoa_ids:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            distance_df = find_distance_to_closest_transport(conn, lsoa_id, transport_lad)
            if distance_df is None: 
                continue
            distance_df_grouped = distance_df.groupby(['lsoa_id', 'transport_index'])
            cur = conn.cursor(pymysql.cursors.DictCursor)
            cur.execute(f"select mt.*, ca.* from method_of_travel_data as mt join car_availability_data as ca on ca.lsoa_id = mt.lsoa_id where ca.lsoa_id = '{lsoa_id}'")
            census_results  = cur.fetchall()[0]
            transport_usage = census_results[transport_type]
            car_availability = census_results['no_car_proportion']
            for idx, distance_df in distance_df_grouped: 
                creation_year = pd.to_datetime(distance_df['creation_date']).dt.year
                distance_df_after = distance_df[pd.to_datetime(distance_df['date_of_transfer']).dt.year >= creation_year]
                distance_df_after['years_after_creation'] = pd.to_datetime(distance_df['date_of_transfer']).dt.year  - creation_year
                avg_dist = np.mean(distance_df['distance'].values)
                median_before = np.median(
                    distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year < creation_year) & distance_df['price'].notna()]['price'].values
                )
                median_after = np.median(
                    distance_df[(pd.to_datetime(distance_df['date_of_transfer']).dt.year >= creation_year) & distance_df['price'].notna()]['price'].values
                )
                pct_inc = (median_after - median_before)/median_before * 100
                
                property_type_counts = distance_df['property_type'].value_counts().to_dict()
                new_build_counts = distance_df['new_build_flag'].value_counts().to_dict()
                merged_counts = property_type_counts | new_build_counts
                combined_df = pd.DataFrame(columns = ['D', 'S', 'T', 'F', 'O', 'Y', 'N'])
                combined_df = pd.concat([combined_df, pd.DataFrame([merged_counts]).fillna(0)], ignore_index=True)
                combined_df = combined_df.fillna(0)
                combined_df[['D', 'S', 'T', 'F', 'O']] = combined_df[['D', 'S', 'T', 'F', 'O']].div(combined_df[['D', 'S', 'T', 'F', 'O']].sum())
                combined_df[['Y', 'N']] = combined_df[['Y','N']].div(combined_df[['Y','N']].sum())
                combined_df = combined_df.fillna(0)
                row = combined_df.iloc[0]
                D_vals.append(row['D'])  
                S_vals.append(row['S'])  
                T_vals.append(row['T']) 
                F_vals.append(row['F'])  
                Y_vals.append(row['Y']) 
                N_vals.append(row['N'])  
                pct_incs.append(pct_inc)
                transport_usage_vals.append(transport_usage)
                car_availability_vals.append(car_availability)
                avg_dists.append(avg_dist)
    features_df = pd.DataFrame({
        'pct_inc': pct_incs,
        'transport_usage': transport_usage_vals, 
        'car_availability': car_availability_vals,
        'avg_dist': avg_dists, 
        'detached': D_vals, 
        'semi_detached': S_vals, 
        'terraced': T_vals, 
        'flats': F_vals, 
        'new_build': Y_vals, 
        })
    return features_df

def find_all_features(conn, lad_id, transport_gdf, transport_type, lad_boundaries, num_lsoas = 5):
    feature_df = {}
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"SELECT unique lsoa_id FROM oa_translation_data where lad_id = '{lad_id}' ORDER BY RAND() LIMIT {num_lsoas}")
    lsoa_ids = list(map(lambda x : x['lsoa_id'], cur.fetchall()))
    transport_lad = find_transport_lad_id_sql(conn, lad_id, transport_type)
    if transport_lad.empty: 
        return
    years_after_creation_vals = np.array([])
    pct_change_vals = np.array([])
    transport_usage_vals = np.array([])
    car_availability_vals = np.array([])
    avg_dists =  np.array([])
    for lsoa_id in lsoa_ids:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            distance_df = find_distance_to_closest_transport(conn, lsoa_id, transport_lad)
            if distance_df is None: 
                continue
            distance_df_grouped = distance_df.groupby(['lsoa_id', 'transport_index'])
            cur = conn.cursor(pymysql.cursors.DictCursor)
            cur.execute(f"select mt.*, ca.* from method_of_travel_data as mt join car_availability_data as ca on ca.lsoa_id = mt.lsoa_id where ca.lsoa_id = '{lsoa_id}'")
            census_results  = cur.fetchall()[0]
            transport_usage = census_results[transport_type]
            car_availability = census_results['no_car_proportion']
            for idx, distance_df in distance_df_grouped: 
                creation_year = pd.to_datetime(distance_df['CreationDateTime']).dt.year
                distance_df_after = distance_df[pd.to_datetime(distance_df['date_of_transfer']).dt.year >= creation_year]
                distance_df_after['years_after_creation'] = pd.to_datetime(distance_df['date_of_transfer']).dt.year  - creation_year
                avg_dist = np.mean(distance_df['distance'].values)
                median_prices = distance_df_after.groupby('years_after_creation')['price'].median().reset_index()
                median_prices['pct_change'] = median_prices['price'].pct_change()
                median_prices = median_prices.dropna()
                years_after_creation_vals = np.concatenate([years_after_creation_vals, median_prices['years_after_creation'].values])
                pct_change_vals = np.concatenate([pct_change_vals, median_prices['pct_change'].values])
                transport_usage_vals = np.concatenate([transport_usage_vals, np.repeat(transport_usage, len(median_prices))])
                car_availability_vals = np.concatenate([car_availability_vals, np.repeat(car_availability, len(median_prices))])
                avg_dists = np.concatenate([avg_dists, np.repeat(avg_dist, len(median_prices))])
    features_df = pd.DataFrame({
        'years_after_creation': years_after_creation_vals,
        'pct_change': pct_change_vals,
        'transport_usage': transport_usage_vals, 
        'car_availability': car_availability_vals,
        'avg_dist': avg_dists
        })
    return features_df
def compute_pairwise_distances(house_gdf, transport_gdf):
    # Extract coordinates as numpy arrays
    coords1 = house_gdf.geometry.apply(lambda geom: (geom.x, geom.y)).to_list()
    coords2 = transport_gdf.geometry.apply(lambda geom: (geom.centroid.x, geom.centroid.y)).to_list()
    # Compute pairwise distances using scipy
    distances = cdist(coords1, coords2, metric='euclidean')

    # Flatten to DataFrame
    distance_df = pd.DataFrame(
        [(i, j, distances[i, j]) for i in range(len(coords1)) for j in range(len(coords2))],
        columns=['house_index', 'transport_index', 'distance']
    )
    return distance_df

def find_closest_points(distance_df):
    # Find the closest point in gdf2 for each point in gdf1
    closest_df = distance_df.loc[distance_df.groupby('house_index')['distance'].idxmin()].reset_index(drop=True)
    return closest_df

def find_transport_lsoa(lsoa_id, transport_df, transport_type, lsoa_boundaries): 
    lad_row = lsoa_boundaries[lsoa_boundaries['LSOA21CD'] == lsoa_id]
    lad_gdf = gpd.GeoDataFrame({'geometry': lad_row.geometry})
    return find_transport_bbox(transport_df, lad_gdf, transport_type)

def find_transport_lsoa_sql(conn, lsoa_id, transport_type): 
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select * from transport_node_data where lsoa_id = '{lsoa_id}' and stop_type = '{transport_type}'")
    lsoa_transport = cur.fetchall()
    return pd.DataFrame(lsoa_transport)


def find_transaction_lsoa(connection, lsoa_id):
    cur = connection.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select * from prices_coordinates_oa_data where lsoa_id = '{lsoa_id}'")
    lsoa_houses = cur.fetchall()
    return pd.DataFrame(lsoa_houses)

def plot_house_price_changes_lsoa(connection, lsoa_id, transport_df):
    cur = connection.cursor(pymysql.cursors.DictCursor)
    houses_df = find_transaction_lsoa(connection, lsoa_id)
    if houses_df.empty:
        return
    transport_df['CreationDateTime'] = pd.to_datetime(transport_df['CreationDateTime'])
    creation_dates = np.unique(transport_df['CreationDateTime'].values)
    house_groups = houses_df.groupby(['street','primary_addressable_object_name', 'secondary_addressable_object_name'])[['price', 'date_of_transfer', 'oa_id']]
    same_houses = {}
    for address, group in house_groups: 
        group = group.drop_duplicates('date_of_transfer')
        if len(group) >= 3: 
            same_houses[address] = group
    same_houses = dict(sorted(same_houses.items(), key=lambda item: len(item[1]), reverse=True))
    keys = list(same_houses.keys())
    sample_size = min(len(keys), 6)
    fig, axs = plt.subplots(3, 2, figsize=(9,9)) 

    for idx in range(min(len(keys), 6)):
        key = keys[idx]
        i, j = divmod(idx, 2)
        house = same_houses[key]
        house['date_of_transfer_datetime'] = pd.to_datetime(house['date_of_transfer'])
        house = house.sort_values(by = 'date_of_transfer_datetime')
        oa_id = house.oa_id.values[0]
        date_of_transfer = mdates.date2num(house.date_of_transfer.values)
        prices = house.price.values
        axs[i, j].plot(date_of_transfer, prices)  
        axs[i, j].scatter(date_of_transfer, prices, alpha = 0.6)
        axs[i, j].set_title(f"{key}", fontsize=8, fontweight='light')
        for date in creation_dates: 
            if date >= np.min(house.date_of_transfer_datetime) and date <= np.max(house.date_of_transfer_datetime):
                date = mdates.date2num(date)
                axs[i, j].axvline(x= date, color='red', linestyle='--', linewidth=1.5, label = 'Creation Date of Transport Facility')
        axs[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        for tick in axs[i, j].get_xticklabels():
            tick.set_rotation(45)  # Rotate tick labels by 45 degrees
        axs[i,j].legend()
    fig.suptitle(f"House Price Trends Over Time in {lsoa_id}")
    plt.tight_layout()
    plt.show()

def plot_house_price_changes_lad(connection, lad_id, transport_df, transport_type, lad_boundaries):
    cur = connection.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select unique lsoa_id from oa_translation_data where lad_id = '{lad_id}'")
    transport_lad = find_transport_lad_id(transport_df, transport_type, lad_id, lad_boundaries)
    lsoas = list(map(lambda x : x['lsoa_id'], cur.fetchall()))
    lsoas_sample = np.random.choice(lsoas, 5, replace = False)
    for lsoa_id in lsoas_sample: 
        plot_house_price_changes_lsoa(connection, lsoa_id, transport_lad)


def plot_distance_to_transport_price_lad(conn, lad_id, lad_boundaries, lsoa_boundaries, transport_df, transport_type):
    distances = []
    lad_row = lad_boundaries[lad_boundaries['LAD21CD'] == lad_id]
    lad_name = lad_row.LAD21NM.values[0]

    avg_col_house_prices = find_avg_lsoa_price_in_lad(conn, lad_id, lad_boundaries)
    avg_col_house_prices_df = pd.DataFrame(avg_col_house_prices)
    transport_nodes = find_transport_lad_id(transport_df, transport_type, lad_id, lad_boundaries)
    lsoas_in_lad = avg_col_house_prices_df.lsoa_id.values
    distances = []
    for lsoa_id in lsoas_in_lad: 
        dist_dict = {}
        lsoa_row = lsoa_boundaries[lsoa_boundaries['LSOA21CD'] == lsoa_id]
        lsoa_geometry = lsoa_row['geometry'].values[0]
        lsoa_centroid = lsoa_geometry.centroid
        for i, row in transport_nodes.iterrows():
            transport_geom = row['geometry']
            transport_atco = row['ATCOCode']
            dist_dict = {'lsoa_id' : lsoa_id, f'distance_to_{transport_atco}': transport_geom.distance(lsoa_centroid)}
            distances.append(dist_dict)
    distances_df = pd.DataFrame(distances)  
    distances_prices_df = distances_df.merge(avg_col_house_prices_df, left_on = 'lsoa_id', right_on = 'lsoa_id')
    distances_prices_df['avg_price'] = distances_prices_df['avg_price'].astype(float)
    distances_prices_df['price_log'] = np.log(distances_prices_df['avg_price'])
    for i, row in transport_nodes.iterrows():
        transport_atco = row['ATCOCode']
        distances = distances_prices_df[f'distance_to_{transport_atco}'].values
        prices = distances_prices_df['price_log'].values
        plt.scatter(distances, prices)
        plt.title(f"Average Log Price of Houses of LSOAs in {lad_name} vs. Distance to {transport_type}")
    return distances, prices
    

def get_lsoa_house_clusters(houses_lsoa): 
    property_types = ['D', 'S', 'T', 'F', 'O']
    property_labels = {'D': 0, 'S': 1, 'T': 2, 'F': 3, 'O': 4}
    new_builds = np.where(houses_lsoa['new_build_flag'] == 'N', 1, 0)
    property_types = [property_labels[property_type] for property_type in houses_lsoa['property_type'].values]
    features = np.column_stack((property_types, new_builds))
    import scipy.cluster.hierarchy as sch
    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(linkage='ward')
    labels = clustering.fit_predict(features)

    # Dendrogram (to visualize the hierarchical structure)
    linkage_matrix = sch.linkage(features, method='ward')
    # plt.figure(figsize=(10, 7))
    # sch.dendrogram(linkage_matrix)
    # plt.show()
    y_threshold = 20 # Cut the dendrogram at y = 5

    # Get the cluster labels for each data point by cutting at the threshold
    clusters = sch.fcluster(linkage_matrix, t=y_threshold, criterion='distance')
    return clusters

def plot_prices_and_clusters(connection, lsoa_id, lsoa_boundaries, building_dfs, transport_type, y_threshold=20): 
    lsoa_row = lsoa_boundaries[lsoa_boundaries['LSOA21CD'] == lsoa_id]
    lsoa_name = lsoa_row.LSOA21NM.values[0]
    lsoa_geom = lsoa_row.geometry.values[0]
    lsoa_bbox = lsoa_row.bbox.values[0]
    houses_lsoa = find_joined_osm_transaction_data_lsoa(connection, lsoa_id, building_dfs)
    if houses_lsoa is None: 
        return
    # houses_lsoa['area_m2'] = houses_lsoa.geometry.area
    houses_lsoa['log_price'] = np.log(houses_lsoa['price'].values)
    houses_lsoa = gpd.GeoDataFrame(houses_lsoa, geometry = gpd.points_from_xy(houses_lsoa['longitude'], houses_lsoa['latitude']))

    clusters = get_lsoa_house_clusters(houses_lsoa)
    houses_lsoa['clusters'] = clusters
    transport_lsoa = find_transport_lsoa_sql(connection, lsoa_id, transport_type)
    if transport_lsoa.empty: 
        return
    transport_gdf = gpd.GeoDataFrame(   
        geometry=gpd.points_from_xy(transport_lsoa['longitude'], transport_lsoa['latitude']),
        crs=houses_lsoa.crs
    )
    fig, ax = plt.subplots()
    price_min, price_max = houses_lsoa['log_price'].min(), houses_lsoa['log_price'].max()
    size_factor = 100  # Adjust this to control the size scaling
    scatter = ax.scatter(
        houses_lsoa.geometry.x, 
        houses_lsoa.geometry.y, 
        c=houses_lsoa['clusters'], 
        s=(houses_lsoa['log_price'] - price_min) / (price_max - price_min) * size_factor, 
        cmap='tab20',  # Color map for clusters
        alpha=0.7,     # Transparency of the dots
        edgecolors='w', # White edge for better contrast
        label='Houses'
    )

    # Add a colorbar for the clusters
    plt.colorbar(scatter, ax=ax, label='Cluster ID')

    # Plot transport nodes
    ax.scatter(
        transport_gdf.geometry.x, 
        transport_gdf.geometry.y, 
        color='red', 
        s=50,   
        label='Transport Nodes'
    )

    ax.set_title(f"Houses in {lsoa_name}: Clusters and Prices", fontsize=16)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    ax.legend()

    plt.show()


def find_median_house_price_change_over_time(conn, lad_id):
    median_house_price = []
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select lad_name from oa_translation_data where lad_id = '{lad_id}'")
    lad_name = cur.fetchall()[0]['lad_name']
    cur.execute(f"select unique lsoa_id from oa_translation_data where lad_id = '{lad_id}'")
    lsoa_ids = list(map(lambda x : x['lsoa_id'], cur.fetchall()))
    for lsoa_id in lsoa_ids:
        transaction_data = find_transaction_lsoa(conn, lsoa_id)
        if transaction_data.empty:
            continue
        transaction_data['year_of_transfer'] = pd.to_datetime(transaction_data['date_of_transfer']).dt.year
        transactions_by_year = transaction_data.groupby('year_of_transfer')
        for year, transaction in transactions_by_year: 
            year_dict = {'lsoa_id': lsoa_id, 'year': year,'median_price': np.median(transaction['price'].values)}
            median_house_price.append(year_dict)
    median_house_price_df = pd.DataFrame(median_house_price)
    median_house_price_df['pct_change'] = median_house_price_df['median_price'].pct_change() * 100 
    return median_house_price_df

def plot_median_house_price_over_time_in_lad(conn, lad_id, transport_gdf, transport_type, lad_boundaries):
    median_house_price_df = find_median_house_price_change_over_time(conn, lad_id)
    grouped_by_lsoa = median_house_price_df.groupby('lsoa_id')
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select lad_name from oa_translation_data where lad_id = '{lad_id}'")
    lad_name = cur.fetchall()[0]['lad_name']
    num_lsoas = min(len(grouped_by_lsoa, 10))
    for lsoa_id, group in itertools.islice(grouped_by_lsoa, num_lsoas): 
        group = group.sort_values(by = 'year')
        years = group['year'].values
        median_prices = group['median_price'].values
        plt.plot(years, median_prices, label = lsoa_id)
        plt.xlabel("Year")
        plt.ylabel("Median Price of Houses in LSOA")
        plt.title(f"Median House Price of LSOAs in {lad_name}")
    transport_df = find_transport_lad_id(transport_gdf, transport_type, lad_id, lad_boundaries)
    creation_years = transport_df.CreationDateTime.dt.year.values
    for year in creation_years: 
        if year >= 2000:
            plt.axvline(x = year, linestyle = '--', color = 'red', label = f'Creation Date of {transport_type}')
    plt.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.5)  
