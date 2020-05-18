"""
Author: Shreya Agrawal
Date: 16 May 2020
Using Folium and H3
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import TimestampedGeoJson
from geojson import Feature, FeatureCollection
import seaborn as sns
import matplotlib.pyplot as plt
from h3 import h3

import branca
import os


class ColorMap:
    def __init__(self, min_v=0, max_v=1, name="", bigger_is_better=True):
        colors_list = ['red', 'yellow', 'green']
        if not bigger_is_better:
            colors_list = colors_list[::-1]

        # self.colormap = branca.colormap.linear.viridis
        # self.colormap = branca.colormap.linear.RdYlGn_11
        self.colormap = branca.colormap.LinearColormap(
            colors_list,
            index=[0.0, 0.5, 1]
        ).scale(min_v, max_v)
        self.colormap.caption = name

    def __call__(self, intensity):
        return self.colormap(intensity)


def norm_col(col):
    """
    Normalizes a Series from 0 to 1
    :param col: pd.Series
    :return:
    """
    return (col - col.min()) / (col.max() - col.min())


def visualize_hexagons(hexagons_df, legends=True, join_muiltiindex_cols=True,
                       polygon_conf_dict={}, folium_map=None, folium_map_config={}):
    """
    Creates a Folium map with hexagons and pop-up legend.
    :param hexagons_df: DataFrame with the H3 index has index.
    :param legends: If 'True', each hexagon will have a pop-up with all the values of its row.
    :param join_muiltiindex_cols: Joins the 'hexagons_df' columns in a single columns, if MultiIndex
    :param polygon_conf_dict: a dict to config the folium.Polygon obj. Ex.:
                {
                    # fill color is True by default, it has to be explicit turned off
                    "color": {"col":"lost_tours", "big_is_better": False, "colormap_legend": "legend", "fill_color": True},
                    "color": {"val":"green"}, # Color can also just be a single value
                    "weight": {"col":"n_bookings", "max": 6, "min":1},
                    "opacity": {"val":0.3},
                    "fill_opacity": {"val":0.15}
                }
                default: opacity: 0.3; fill_opacity: 0.15; weight: 0.5; color: green
    :param folium_map: The folium map obj to had the Hexagons to. If None a new map is created
    :param folium_map_config: default = {"zoom_start": 11,"tiles": 'cartodbpositron', "location": avg_hexagon_location}
                location: The initial lat, long to start the map at. If None tha Avg center of the hexagons will be used.
                zoom_start: the 'folium.Map()' zoom_start value. default 11
                tiles: the 'folium.Map()' tiles value. default 'cartodbpositron'
    :return: Folium map obj
    """

    hexagons_df = hexagons_df.copy()

    hexagons = hexagons_df.index.values
    n_hexs = len(hexagons)
    add_color_map_to_map = False

    # Hexagons popup Legends
    if legends is None or legends is False:
        legends = np.array(n_hexs * [None])
    elif legends is True:
        if join_muiltiindex_cols and type(hexagons_df.columns) == pd.MultiIndex:
            hexagons_df.columns = ["-".join(c) for c in hexagons_df.columns]
        hexagons_dict = hexagons_df.to_dict("index")
        legends = [f"{idx}: {hexagons_dict[idx]}" for idx in hexagons]

    # processing Polygon Propreties
    # Adding default Polygon configs
    polygon_conf_dict.setdefault("opacity", {"val": 0.3})
    polygon_conf_dict.setdefault("fill_opacity", {"val": 0.15})
    polygon_conf_dict.setdefault("weight", {"val": 0.5})
    polygon_conf_dict.setdefault("color", {"val": "green"})

    all_poly_props_df = pd.DataFrame()
    for col_name, conf in polygon_conf_dict.items():
        if "col" in conf:
            poly_prop_values = hexagons_df[conf["col"]]
            # Normalize
            if set(("min", "max")).issubset(conf):
                poly_prop_values = (conf["min"] + norm_col(poly_prop_values) * (conf["max"] - conf["min"])).values
        elif "val" in conf:
            poly_prop_values = len(hexagons_df) * [conf["val"]]
        #     else:
        #         raise Exception("No 'col' or 'val' key found! :(")
        # Processing colors
        if col_name == "color":
            if "col" in conf:
                big_is_better = conf["big_is_better"] if "big_is_better" in conf else True
                cm_legend = conf["colormap_legend"] if "colormap_legend" in conf else str(conf["col"]).replace("'", "")
                colormap = ColorMap(np.min(poly_prop_values), np.max(poly_prop_values), cm_legend, big_is_better)
                poly_prop_values = [colormap(ci) for ci in poly_prop_values]
                add_color_map_to_map = True
            # Adds fill color by default, it has to be explicit turned off
            if ("fill_color" in conf and conf["fill_color"]) or ("fill_color" not in conf):
                all_poly_props_df["fill_color"] = poly_prop_values

        all_poly_props_df[col_name] = poly_prop_values

    polys_config = list(all_poly_props_df.to_dict("index").values())

    # Initial Location
    if "location" not in folium_map_config:
        location = np.mean([h3.h3_to_geo(h3_id) for h3_id in hexagons], axis=0)
        folium_map_config["location"] = location

    # Creates Folium map
    if folium_map is None:
        folium_map_config.setdefault("zoom_start", 11)
        folium_map_config.setdefault("tiles", 'cartodbpositron')
        m = folium.Map(**folium_map_config)
    else:
        m = folium_map

    # adding polygons
    for hex_id, leg, poly_conf in zip(hexagons, legends, polys_config):
        locations = h3.h3_to_geo_boundary(hex_id)
        folium.Polygon(locations, popup=leg, **poly_conf).add_to(m)
    # adds the colormap legend to the map
    if add_color_map_to_map:
        m.add_child(colormap.colormap)
        colormap.colormap.add_to(m)
    return m

def downscale_h3(time_win_df, agg_brothers, downscale_size=2, h3_index_col = "h3_index"):
    # tODO: 2nd for loop should be a while with selected res at top
    # Only with this number of brother the hexagons will be scaled (max = 7)
    n_min_brothers_to_scale = 5

    time_win_h3 = time_win_df.reset_index()
    time_win_h3["h3_res"] = time_win_h3[h3_index_col].apply(h3.h3_get_resolution)
    downscale_resulutions = range(time_win_h3["h3_res"].min(),
                                  time_win_h3["h3_res"].min()-downscale_size, -1)

    for child_h3_res_depth, downscale_res in enumerate(downscale_resulutions):
        print("Auto downscale h3 resolution:", downscale_res)
        for idx, row in time_win_h3.iterrows():
            # Once time_win_h3 indexs get changed during the loop and the time_win_h3.iterrows()
            # is a copy of the rows, they might not exist
            if idx not in time_win_h3.index:
                continue
                
            h3idx = row[h3_index_col]
            cell_res = row["h3_res"]
            # If its a different res than the one we re trying to downscale, skips
            if cell_res != downscale_res:
                continue
            
            parent = h3.h3_to_parent(h3idx, cell_res - 1)
            # finding all the brother cells
            brother_cells = list(h3.h3_to_children(parent, cell_res))
            # dont scale if there is less than X brothers
            if time_win_h3[h3_index_col].isin(brother_cells).sum() < n_min_brothers_to_scale:
                continue
                
            # finding all the children cells
            for childh3res in range(1,child_h3_res_depth+1):
                brother_cells.extend(list(h3.h3_to_children(parent, cell_res + childh3res)))

            brothers_df = time_win_h3[time_win_h3[h3_index_col].isin(brother_cells)]
            
            agg_result = agg_brothers(brothers_df)

            if agg_result is False:
                continue
            # set the cols to the parent values
            agg_result[h3_index_col] = parent
            agg_result["h3_res"] = cell_res - 1
            time_win_h3.loc[idx] = agg_result
            # drop the rest of the brothers
            time_win_h3.drop([i for i in brothers_df.index if i != idx], inplace=True)

    return time_win_h3.set_index(h3_index_col)

def agg_brothers(brothers_df):
    # Brothers with less than 3 will be scaled to their parent
    max_brothers_sum = 3
    brothers_sum = brothers_df[['count', 'supply_value']].sum()

    if any(brothers_sum > max_brothers_sum):
        return False
    
    return brothers_sum


def __create_time_geojson_features(df, time_col_str):
    features = []
    for _, row in df.iterrows():
        feature = {
            'type': 'Feature',
            'geometry': {
                'type':'Polygon',
                'coordinates': [[[e2, e1] for e1,e2 in row['hex_pols'] + [row['hex_pols'][0]]]]
            },
            'properties': {
                'time': row[time_col_str], #row['time_window_15'].strftime('%Y-%m-%dT%H:%M:%S'),
                'style': {'color' : row['color_bin']},
                'iconstyle':{
                    'fillColor': row['color_bin'],
                    'fillOpacity': 0.8,
                    'stroke': 'true'
                }
            }
        }
        features.append(feature)
    return features

def __make_map(features, initial_coords):
    coords = initial_coords #[52.24, 21.02]
    map_ = folium.Map(location=coords, control_scale=True, zoom_start=8)

    TimestampedGeoJson(
        {'type': 'FeatureCollection',
        'features': features}
        , period = 'PT15M'
        , duration = 'PT1M'
        , add_last_point=True
        , auto_play=False
        , loop=False
        , loop_button=True
        , time_slider_drag_update=True
    ).add_to(map_)

    return map_

def folium_dynamic_hexagons(df, initial_coords, h3_col_str, time_col_str, agg_col_str):

    # add H3 hexagons polygons
    h3_unique = df['h3_address_7'].drop_duplicates()
    hex_pols = h3_unique.apply(h3.h3_to_geo_boundary)
    h3_hex = pd.DataFrame({'h3_address_7': h3_unique, 'hex_pols':hex_pols})
    df_ = pd.merge(df, h3_hex, how='left')

    # add color to agg counts
    colors = np.array(['#053061','#2166ac','#4393c3','#92c5de','#d1e5f0','#fddbc7','#f4a582','#d6604d','#b2182b','#67001f'])
    bins = np.linspace(df_[agg_col_str].min(), df_[agg_col_str].max(), len(colors))
    display(bins)
    sns.palplot(sns.color_palette(colors))
    plt.show()
    df_['color_bin'] = colors[np.digitize(df_[agg_col_str], bins, right=True)]


    features = __create_time_geojson_features(df_, time_col_str)
    print(len(features))
    return [features, __make_map(features,initial_coords)]


def folium_static_hexagons(df, h3_index_str='h3_index', h3_hex_str='h3_hexagon'):
    feature_collection = []
    for index, row in df.iterrows():
        feature = {
            'type':'Feature',
            'geometry': {
                   'type': 'Polygon',
                    'coordinates': [[[elem[1], elem[0]] for elem in row[h3_hex_str] + [row[h3_hex_str][0]]]]
            }
        }
        feature_collection.append(feature)

    f_c = FeatureCollection(feature_collection)

    initial_coords = h3.h3_to_geo(df[h3_index_str][0])
    map_ = folium.Map(location=initial_coords, control_scale=True, zoom_start=9)

    f_json = folium.GeoJson(f_c)

    map_.add_child(f_json)
    return map_
