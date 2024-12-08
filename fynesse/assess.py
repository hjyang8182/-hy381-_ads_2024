import pymysql.cursors
from .config import *
from .access import *
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon
import pymysql
import geopandas as gpd
import seaborn as sns
from geopy.distance import geodesic
import ast
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import random
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
    merged_on_coord = gpd.sjoin_nearest(transactions_not_merged, osm_df, max_distance = 0.005)
    merged_on_coord = merged_on_coord.drop_duplicates(subset = 'left_index')
    merged_on_coord = merged_on_coord.drop(columns = 'left_index')

    cols = ['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type', 'primary_addressable_object_name', 'secondary_addressable_object_name', 'street', 'latitude', 'longitude', 'db_id', 'geometry']
    merged_on_addr = merged_on_addr[cols]
    merged_on_coord = merged_on_coord[cols]
    full_merged = pd.concat([merged_on_addr, merged_on_coord], ignore_index = True)
    
    full_merged = gpd.GeoDataFrame(full_merged, geometry = 'geometry')
    full_merged['price_log'] = np.log(full_merged['price'])
    return full_merged

def find_transport_bbox(transport_gdf, lad_gdf, transport_type): 
    type_codes = []
    if transport_type == 'BUS': 
        type_codes = ('BCT', 'BCS', 'BCQ', 'BST', 'BCE', 'BCP')
    elif transport_type == 'SUB': 
        type_codes = ('PLT', 'MET', 'TMU')
    elif transport_type == 'RAIL': 
        type_codes = ('RPLY', 'RLY')
    elif transport_type == 'AIR':
        type_codes = ('AIR', 'GAT')
    else: 
       pass
    # Transport node data seems to be missing a lot of values
    # cur = conn.cursor(pymysql.cursors.DictCursor)
    # cur.execute(f"select * from transport_node_data where longitude between {south} and {north} and latitude between {east} and {west} and stop_type in {type_codes}")
    transport_gdf = gpd.sjoin(transport_gdf, lad_gdf, predicate = 'within')
    transport_gdf = transport_gdf[np.isin(transport_gdf['StopType'], type_codes)]
    transport_gdf = gpd.GeoDataFrame(transport_gdf)
    return transport_gdf

def find_transaction_bbox(conn, bbox): 
    west, south, east, north = bbox
    cur = conn.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select * from prices_coordinates_oa_data where latitude between {south} and {north} and longitude between {west} and {east} and date_of_transfer >= 2020-01-01")
    transaction_df = pd.DataFrame(cur.fetchall())
    return transaction_df

def find_residential_buildings(conn, lad_id, building_dfs): 
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

def plot_lad_prices(conn, lad_id, building_dfs, lad_boundaries, transport_gdf, transport_type):
    """
        transport type: 
            'BUS' :  bus stops, station entrances
            'SUB' : tube, tram, underground platform entrances
            'RAIL' : railway stations 
            'AIR' : airport, air access areas
    """ 
    buildings_gdf = find_residential_buildings(conn, lad_id, building_dfs)
    lad_row = lad_boundaries[lad_boundaries['LAD21CD'] == lad_id]
    lad_name = lad_row.LAD21NM.values[0]
    lad_geom = lad_row.geometry.values[0]
    lad_bbox = lad_row.bbox.values[0]
    lad_gdf = gpd.GeoDataFrame({'geometry': lad_row.geometry})

    transport_gdf = find_transport_bbox(transport_gdf, lad_gdf, transport_type)
    house_transactions = find_transaction_bbox(conn, lad_bbox)
    osm_prices_merged = join_osm_transaction_data(buildings_gdf, house_transactions)

    osm_prices_merged = gpd.sjoin(osm_prices_merged, lad_gdf, predicate = 'within')
    fig, ax = plt.subplots()
    lad_gdf.plot(ax = ax, facecolor = 'white', edgecolor = 'dimgray')
    osm_prices_merged.plot(column = 'price_log', ax = ax, legend = True, cmap = 'viridis')
    transport_gdf.plot(ax = ax, color = 'red', markersize = 10)

    # custom_patch = mpatches.Patch(color='red', label='Transport Facilities')
    # ax.legend(handles=[custom_patch], title="Legend")
    plt.title(f"log Price of Houses in {lad_name}")
    plt.show()

def find_transport_lsoa(connection, lsoa_id): 
    cur = connection.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select * from transport_node_data where lsoa_id = '{lsoa_id}'")
    oa_df = cur.fetchall()
    return pd.DataFrame(oa_df)

def find_transaction_lsoa(connection, lsoa_id):
    cur = connection.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select * from prices_coordinates_oa_data where lsoa_id = '{lsoa_id}'")
    lsoa_houses = cur.fetchall()
    return pd.DataFrame(lsoa_houses)

def plot_house_price_changes(connection, lsoa_id):
    cur = connection.cursor(pymysql.cursors.DictCursor)
    houses_df = find_transaction_lsoa(connection, lsoa_id)
    transport_df = find_transport_lsoa(connection, lsoa_id)
    creation_dates = np.unique(transport_df.creation_date.values)
    house_groups = houses_df.groupby(['street','primary_addressable_object_name', 'secondary_addressable_object_name'])[['price', 'date_of_transfer', 'oa_id']]
    same_houses = {}
    for address, group in house_groups: 
        if len(group) > 1: 
            same_houses[address] = group
    keys = list(same_houses.keys())
    same_houses_sample = random.sample(keys, 9)
    fig, axs = plt.subplots(3, 3, figsize=(12, 12)) 
    for i in range(3):
        for j in range(3):
            key = same_houses_sample[i*j]
            house = same_houses[key]
            oa_id = house.oa_id.values[0]
            date_of_transfer = mdates.date2num(house.date_of_transfer.values)
            prices = house.price.values
            axs[i, j].plot(date_of_transfer, prices)  
            axs[i, j].set_title(f"{key}", fontsize=8, fontweight='light')
            for date in creation_dates: 
                if date >= np.min(house.date_of_transfer) and date <= np.max(house.date_of_transfer):
                    date = mdates.date2num(date)
                    axs[i, j].axvline(x= date, color='red', linestyle='--', linewidth=1.5)
            axs[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            for tick in axs[i, j].get_xticklabels():
                tick.set_rotation(45)  # Rotate tick labels by 45 degrees
    plt.tight_layout()

def find_dist_house_corr_lsoa(connection, lsoa_id, transport_lsoa, house_lsoa):
    avg_distances = np.array([])
    prices = np.array([])
    cur = connection.cursor(pymysql.cursors.DictCursor)
    cur.execute(f"select lsoa_name from oa_boundary data where lsoa_id = {lsoa_id}")
    lsoa_name = cur.fetchall()[0]['lsoa_name']
    houses_lsoa = find_transaction_lsoa(lsoa_name)
    houses_lsoa = gpd.GeoDataFrame(houses_lsoa, geometry = gpd.points_from_xy(houses_lsoa['longitude'], houses_lsoa['latitude']))
    houses_lsoa['avg_distance'] = houses_lsoa.geometry.apply(lambda house: find_avg_distance(house, transport_lsoa))
    avg_distances = np.append(avg_distances, houses_lsoa['avg_distance'].values)
    prices = np.append(prices, houses_lsoa['price'].values)
    plt.figure()
    plt.scatter(avg_distances, prices)
    return avg_distances, prices

def find_avg_distance(house_point, transport_df):
    distances = transport_df.distance(house_point)
    return distances.mean()

def get_lsoa_house_clusters(houses_lsoa): 
    property_types = ['D', 'S', 'T', 'F', 'O']
    property_labels = {'D': 0, 'S': 1, 'T': 2, 'F': 3, 'O': 4}
    houses_oa = houses_lsoa[houses_lsoa['area_m2'].notna()]
    areas = houses_oa['area_m2'].values
    new_builds = np.where(houses_oa['new_build_flag'] == 'N', 1, 0)
    property_types = [property_labels[property_type] for property_type in houses_oa['property_type'].values]
    features = np.column_stack((property_types, new_builds, areas))
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

def plot_prices_and_clusters(connection, lsoa_id, lsoa_boundaries, y_threshold=20): 
    lsoa_row = lsoa_boundaries[lsoa_boundaries['LSOA21CD'] == lsoa_id]
    lsoa_name = lsoa_row.LAD21NM.values[0]
    lsoa_geom = lsoa_row.geometry.values[0]
    lsoa_bbox = lsoa_row.bbox.values[0]

    houses_lsoa = find_transaction_bbox(connection, lsoa_bbox)
    houses_lsoa = gpd.GeoDataFrame(houses_lsoa, geometry = gpd.points_from_xy(houses_lsoa['longitude'], houses_lsoa['latitude']))

    clusters = get_lsoa_house_clusters(houses_lsoa)
    houses_lsoa['clusters'] = clusters

    transport_lsoa = find_transport_bbox(connection, lsoa_bbox)
    transport_nodes_coords = list(map(lambda n: (n['longitude'], n['latitude']), transport_lsoa))

    fig, ax = plt.subplots(1, 2)
    houses_lsoa.plot(column = 'clusters', ax = ax[0], legend = True, cmap = 'tab20')
    houses_lsoa.plot(column = 'price', ax = ax[1], legend = True, cmap = 'YlOrRd')
    for coord in transport_nodes_coords:
        ax[0].scatter(coord[0], coord[1], color = 'red')
    ax[0].set_title(f"Clusters for houses in {lsoa_name}")
    ax[1].set_title(f"Prices and Transport Nodes for Houses in {lsoa_name}")