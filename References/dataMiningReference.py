import pandas as pd
import networkx as nx
import osmnx as ox
import folium.plugins
from folium import FeatureGroup, LayerControl, Map, Marker, Icon, PolyLine
import branca.colormap as cm
from folium.plugins import HeatMap
from IPython.display import display
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt

# I am adding columns to a copied selection of a Dataframe and I don't want warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Load the data
movement_data = pd.read_csv('Cambridge_gowalla.csv')

"""Part 1"""


def part_1(data):
    # These are the users we are interested in
    users = [75027, 102829]

    users_data = data.loc[data['User_ID'].isin(users)]

    users_data['timestamp'] = 0

    # Convert the date to a timestamp so it may be visualised
    users_data['timestamp'] = users_data['date'].apply(
        lambda x: time.mktime(datetime.datetime.strptime(x, '%d/%m/%Y').timetuple()) - np.min(users_data['timestamp']))

    # Dict to store user geoms
    user_points = {}
    # Loop through users and store their [lat, lon, timestamp] in the dict
    for i in users:
        user_data = users_data.loc[users_data['User_ID'] == i]
        user_points[i] = [i for i in zip(user_data['lat'], user_data['lon'], users_data['timestamp'])]

    # Set up a colormap so the timestamp can be visualised
    colormap = cm.LinearColormap(colors=['red', 'blue'],
                                 index=[np.min(users_data['timestamp']), np.max(users_data['timestamp'])],
                                 vmin=np.min(users_data['timestamp']), vmax=np.min(users_data['timestamp']))

    # Initialise the map object
    my_map = Map([np.mean(data['lat']), np.mean(data['lon'])], zoom_start=12.5)

    # Colour for each user
    colors = ['green', 'orange']

    # Loop through each user in the dict
    for i, (k, v) in enumerate(user_points.items()):
        color = colors[i]
        # Loop through the points/timestamps
        for p in v:
            folium.Circle(
                location=p[0:2],
                radius=80,
                fill=True,
                fill_color=color,
                color=colormap(p[2]),
                fill_opacity=1
            ).add_to(my_map)

    my_map.save('my_map_1_1.html')


part_1(movement_data)


"""Part 2"""


def part_2(data):
    # Each user now has a data associated with them
    users = {
        75027: '30/01/2010',
        102829: '11/05/2010'
    }

    users_data = data.loc[data['User_ID'].isin(users)]

    # We can start with the Dataframe used previously to get our new subset
    users_data = users_data.loc[(((users_data['User_ID'] == list(users.keys())[0]) &
                                  (users_data['date'] == list(users.values())[0])) |
                                 ((users_data['User_ID'] == list(users.keys())[1]) &
                                  (users_data['date'] == list(users.values())[1])))]

    G = ox.graph_from_bbox(np.max(users_data['lat']) + 0.02, np.min(users_data['lat']) - 0.02,
                           np.max(users_data['lon']) + 0.02, np.min(users_data['lon']) - 0.02)

    ox.plot_graph(G)

    # # Add a date timestamp
    # users_data['timestamp'] = 0
    #
    # users_data['timestamp'] = users_data['date'].apply(
    #     lambda x: time.mktime(datetime.datetime.strptime(x, '%d/%m/%Y').timetuple()))
    #
    # # Sort values by timestamp and time so that we look at routes as consecutive sets of points
    # users_data = users_data.sort_values(['timestamp', 'Time'])
    #
    # routes, origin_points, dest_points = {}, [], []  # Some empty iterators to take data
    #
    # dates = {k: 0 for k in users_data['date'].unique()}  # a dictionary to hold date length summations
    #
    # # Getting the latlon from the dataa
    # for user in list(users_data['User_ID'].unique()):  # For each user
    #     for j, point in enumerate(
    #             list(users_data.loc[users_data['User_ID'] == user].iterrows())[:-1]):  # For each point
    #         # point is in form of [user, [date, time, etc.]]
    #         origin_point = (point[1]['lon'], point[1]['lat'])  # its lon-lat in the df for some reason
    #         dest_point = (list(users_data.loc[users_data['User_ID'] == user].iterrows())[j + 1][1][5],
    #                       list(users_data.loc[users_data['User_ID'] == user].iterrows())[j + 1][1][4])
    #         origin_points.append(origin_point)
    #         dest_points.append(dest_point)
    #         origin_node = ox.get_nearest_node(G, origin_point)
    #         dest_node = ox.get_nearest_node(G, dest_point)
    #         route = nx.shortest_path(G, origin_node, dest_node, weight='length')
    #         length = nx.shortest_path_length(G, origin_node, dest_node, weight='length')
    #         routes[str(user) + ' route ' + str(j + 1)] = [route, length]
    #         dates[point[1]['date']] += length
    #
    # def route_with_points(i, routes=routes, origin_points=origin_points, dest_points=dest_points):
    #     # set show and close to false so we can add to plot and not have it close automatically
    #     fig, ax = ox.plot_graph_route(G, list(routes.values())[i][0], show=False, close=False,
    #                                   origin_point=origin_points[i],
    #                                   destination_point=dest_points[i])
    #     ax.scatter(origin_points[i][0], origin_points[i][1], c='green')
    #     ax.scatter(dest_points[i][0], dest_points[i][1], c='red')
    #     plt.show()
    #
    # route_with_points(0)
    #
    # lengths = [i[1] for i in routes.values()]
    #
    # max_displacement = np.max(lengths)
    #
    # avg_displacement = np.mean(lengths)


part_2(movement_data)