
import os
import sys
import pickle
import argparse
#import sklearn
from data_management.read_csv import *
#from numba import njit      # specifically for finding the starting index

# dataset location: "E:/Research/Implementation/HighD_preprocessing/data/01_tracks.csv"
def create_args(path, track_meta_path):
    parser = argparse.ArgumentParser(description="ParameterOptimizer")
    # --- Input paths ---
    parser.add_argument('--input_path', default=path, type=str,
                        help='CSV file of the tracks')
    #parser.add_argument('--input_path_02_tracks', default="E:/Research/Implementation/project_1/data/02_tracks.csv", type=str,
    #                    help='CSV file of the tracks')
    parser.add_argument('--input_static_path', default=track_meta_path,
                       type=str, help='Static track meta data file for each track')
    #parser.add_argument('--input_meta_path', default=track_meta_path,
    #                    type=str,help='Static meta data file for the whole video')
    parser.add_argument('--pickle_path', default="../data/01.pickle", type=str,
                        help='Converted pickle file that contains corresponding information of the "input_path" file')

    '''
    # --- Settings ---
    parser.add_argument('--visualize', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='True if you want to visualize the data.')
    parser.add_argument('--background_image', default="E:/Research/Implementation/HighD_preprocessing/data/01_highway.png", type=str,
                          help='Optional: you can specify the correlating background image.')
                          


    # --- Visualization settings ---
    parser.add_argument('--plotBoundingBoxes', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to plot the bounding boxes or not.')
    parser.add_argument('--plotDirectionTriangle', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to plot the direction triangle or not.')
    parser.add_argument('--plotTextAnnotation', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to plot the text annotation or not.')
    parser.add_argument('--plotTrackingLines', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to plot the tracking lines or not.')
    parser.add_argument('--plotClass', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to show the class in the text annotation.')
    parser.add_argument('--plotVelocity', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to show the class in the text annotation.')
    parser.add_argument('--plotIDs', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: decide whether to show the class in the text annotation.')

    '''
    # --- I/O settings ---
    parser.add_argument('--save_as_pickle', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Optional: you can save the tracks as pickle.')
    parsed_arguments = vars(parser.parse_args())
    return parsed_arguments



def get_dataset(path, track_meta_path):
    # get the tracks
    created_arguments = create_args(path, track_meta_path)
    print("Try to find the saved pickle file for better performance.")
    # Read the track csv and convert to useful format
    if os.path.exists(created_arguments["pickle_path"]):
        with open(created_arguments["pickle_path"], "rb") as fp:
            tracks = pickle.load(fp)
        print("Found pickle file {}.".format(created_arguments["pickle_path"]))
    else:
        print("Pickle file not found, csv will be imported now.")
        tracks = read_track_csv(created_arguments)
        #print("Finished importing the pickle file.")
        track_meta = read_static_info(created_arguments)    # for track meta information

    if created_arguments["save_as_pickle"] and os.path.exists(created_arguments["pickle_path"]):
        print("Save tracks to pickle file.")

        with open(created_arguments["pickle_path"], "wb") as fp:
            pickle.dump(tracks, fp)
    '''
    # Read the static info
    try:
        static_info = read_static_info(created_arguments)
    except:
        print("The static info file is either missing or contains incorrect characters.")
        sys.exit(1)
    '''
    return tracks, track_meta #,static_info

# prepare a build_input() function



def get_input_vector(tracks):
    # for a single track
    num_agent_features = 12
    num_neighbor_features = 6
    num_neighbors = 8
    num_total_row = num_agent_features + num_neighbor_features * num_neighbors  # number of rows in a given track
    inputs = []

    # stack the arrays
    for id in range(len(tracks)):
        t = tracks[id]
        track_id = t['id']
        num_frames = t['frame'].size
        start_frame = t['frame'][0] # frame number of the starting of this track
        inp_v = np.zeros((num_total_row, num_frames))   #input vector
        fill_inp_v_start = 0        # for the vehicle (agent) starting row for putting the features
        fill_inp_v_end = num_agent_features         #end row
        inp_v[fill_inp_v_start:fill_inp_v_end] = np.stack((t['center_x'], t['center_y'],t['xVelocity'], t['yVelocity'], t['xAcceleration'], t['yAcceleration'],
                 t['frontSightDistance'], t['backSightDistance'], t['thw'], t['ttc'], t['dhw'], t['precedingXVelocity']))
        #num_frame  = tracks[id]['frame'].size

        #get neighbor's info
        preceding_nbr = tracks[id]['precedingId']
        following_nbr = tracks[id]['followingId']
        leftFollwoing_nbr = tracks[id]['leftFollowingId']
        lefAlongside_nbr = tracks[id]['leftAlongsideId']
        leftPreceding_nbr = tracks[id]['leftPrecedingId']
        rightFollowing_nbr = tracks[id]['rightFollowingId']
        rightAlongside_nbr = tracks[id]['rightAlongsideId']
        rightPreceding_nbr = tracks[id]['rightPrecedingId']

        #inp_v = np.vstack((inp_v, np.zeros((48, inp_v.size)))) # 8 neighboring vehicles * number of features for each neighbor(=6)

        '''
        # build preceding_nbr feature table
        preceding_nbr_features = np.zeros((num_neighbor_features, inp_v.shape[1]))
        for i in range(preceding_nbr.shape[0]):   # or inp_v.shape[1]
            if preceding_nbr[i] != 0:
                track_ID = preceding_nbr[i] - 1
                nt = tracks[track_ID]     # array index starts at 0, so is the index of tracks. e.g. tracks of ID 12 is available in tracks[11]

                preceding_nbr_features[:,i] = np.array((nt['center_x'][i-1], nt['center_y'][i-1], nt['xVelocity'][i-1], nt['yVelocity'][i-1], nt['xAcceleration'][i-1], nt['yAcceleration'][i-1])).transpose()
        fill_inp_v_start = num_agent_features
        fill_inp_v_end = num_agent_features + 6
        inp_v[fill_inp_v_start:fill_inp_v_end] = preceding_nbr_features
        '''
        ## another way
        preceding_nbr_features = np.zeros((num_neighbor_features, inp_v.shape[1])) # inp_v.shape[1] : length of trajectory/track of the agent
        unique_nbr = np.unique(preceding_nbr) # list of preceding neighbors throughout the track
        unique_nbr_indices = [] # list the indices at which the unique neighbors appears
        for i in range(len(unique_nbr)):
            unique_nbr_indices.append(np.where(preceding_nbr==unique_nbr[i]))
            if unique_nbr[i] !=0:
                nt = tracks[unique_nbr[i]-1]  # -1 since tracks serial number starts from 0
                idx = unique_nbr_indices[i]
                for j, value in enumerate(idx[0]):      # value is the index number of the agent where this nbr appears
                    #frame_ref = value + start_frame    #reference frame
                    frame_ref = t['frame'][value]       # what is the corresponding frame number for this index
                    nbr_start_frame = nt['frame'][0]
                    diff = frame_ref - nbr_start_frame
                    preceding_nbr_features [:, value] = np.array((nt['center_x'][diff],
                                                                  nt['center_y'][diff],
                                                                  nt['xVelocity'][diff],
                                                                  nt['yVelocity'][diff],
                                                                  nt['xAcceleration'][diff],
                                                                  nt['yAcceleration'][diff])).transpose()

        fill_inp_v_start = num_agent_features
        fill_inp_v_end = num_agent_features + 6
        inp_v[fill_inp_v_start:fill_inp_v_end] = preceding_nbr_features

        # build following_nbr feature table
        following_nbr_features = np.zeros(
            (num_neighbor_features, inp_v.shape[1]))  # inp_v.shape[1] : length of trajectory/track of the agent
        unique_nbr = np.unique(following_nbr)  # list of preceding neighbors throughout the track
        unique_nbr_indices = []  # list the indices at which the unique neighbors appears
        for i in range(len(unique_nbr)):
            unique_nbr_indices.append(np.where(following_nbr == unique_nbr[i]))
            if unique_nbr[i] != 0:
                nt = tracks[unique_nbr[i] - 1]  # -1 since tracks serial number starts from 0
                idx = unique_nbr_indices[i]
                for j, value in enumerate(idx[0]):  # value is the index number of the agent where this nbr appears
                    frame_ref = t['frame'][value]  # what is the corresponding frame number for this index
                    nbr_start_frame = nt['frame'][0]
                    diff = frame_ref - nbr_start_frame  # reference frame
                    following_nbr_features[:, value] = np.array((nt['center_x'][diff],
                                                                 nt['center_y'][diff],
                                                                 nt['xVelocity'][diff],
                                                                 nt['yVelocity'][diff],
                                                                 nt['xAcceleration'][diff],
                                                                 nt['yAcceleration'][diff])).transpose()

        fill_inp_v_start = num_agent_features + 6
        fill_inp_v_end = num_agent_features + 12
        inp_v[fill_inp_v_start:fill_inp_v_end] = following_nbr_features


        # build leftFollwoing_nbr feature table
        leftFollwoing_nbr_features = np.zeros(
            (num_neighbor_features, inp_v.shape[1]))  # inp_v.shape[1] : length of trajectory/track of the agent
        unique_nbr = np.unique(leftFollwoing_nbr)  # list of preceding neighbors throughout the track
        unique_nbr_indices = []  # list the indices at which the unique neighbors appears
        for i in range(len(unique_nbr)):
            unique_nbr_indices.append(np.where(leftFollwoing_nbr == unique_nbr[i]))
            if unique_nbr[i] != 0:
                nt = tracks[unique_nbr[i] - 1]  # -1 since tracks serial number starts from 0
                idx = unique_nbr_indices[i]
                for j, value in enumerate(idx[0]):  # value is the index number of the agent where this nbr appears
                    frame_ref = t['frame'][value]  # what is the corresponding frame number for this index
                    nbr_start_frame = nt['frame'][0]
                    diff = frame_ref - nbr_start_frame
                    leftFollwoing_nbr_features[:, value] = np.array((nt['center_x'][diff],
                                                                 nt['center_y'][diff],
                                                                 nt['xVelocity'][diff],
                                                                 nt['yVelocity'][diff],
                                                                 nt['xAcceleration'][diff],
                                                                 nt['yAcceleration'][diff])).transpose()

        fill_inp_v_start = num_agent_features + 12
        fill_inp_v_end = num_agent_features + 18
        inp_v[fill_inp_v_start:fill_inp_v_end] = leftFollwoing_nbr_features

        # build lefAlongside_nbr feature table
        lefAlongside_nbr_features = np.zeros(
            (num_neighbor_features, inp_v.shape[1]))  # inp_v.shape[1] : length of trajectory/track of the agent
        unique_nbr = np.unique(lefAlongside_nbr)  # list of preceding neighbors throughout the track
        unique_nbr_indices = []  # list the indices at which the unique neighbors appears
        for i in range(len(unique_nbr)):
            unique_nbr_indices.append(np.where(lefAlongside_nbr == unique_nbr[i]))
            if unique_nbr[i] != 0:
                nt = tracks[unique_nbr[i] - 1]  # -1 since tracks serial number starts from 0
                idx = unique_nbr_indices[i]
                for j, value in enumerate(idx[0]):  # value is the index number of the agent where this nbr appears
                    frame_ref = t['frame'][value]  # what is the corresponding frame number for this index
                    nbr_start_frame = nt['frame'][0]
                    diff = frame_ref - nbr_start_frame
                    lefAlongside_nbr_features[:, value] = np.array((nt['center_x'][diff],
                                                                     nt['center_y'][diff],
                                                                     nt['xVelocity'][diff],
                                                                     nt['yVelocity'][diff],
                                                                     nt['xAcceleration'][diff],
                                                                     nt['yAcceleration'][diff])).transpose()
        fill_inp_v_start = num_agent_features + 18
        fill_inp_v_end = num_agent_features + 24
        inp_v[fill_inp_v_start:fill_inp_v_end] = lefAlongside_nbr_features

        # build leftPreceding_nbr feature table
        leftPreceding_nbr_features = np.zeros(
            (num_neighbor_features, inp_v.shape[1]))  # inp_v.shape[1] : length of trajectory/track of the agent
        unique_nbr = np.unique(leftPreceding_nbr)  # list of preceding neighbors throughout the track
        unique_nbr_indices = []  # list the indices at which the unique neighbors appears
        for i in range(len(unique_nbr)):
            unique_nbr_indices.append(np.where(leftPreceding_nbr == unique_nbr[i]))
            if unique_nbr[i] != 0:
                nt = tracks[unique_nbr[i] - 1]  # -1 since tracks serial number starts from 0
                idx = unique_nbr_indices[i]
                for j, value in enumerate(idx[0]):  # value is the index number of the agent where this nbr appears
                    frame_ref = t['frame'][value]  # what is the corresponding frame number for this index
                    nbr_start_frame = nt['frame'][0]
                    diff = frame_ref - nbr_start_frame
                    leftPreceding_nbr_features[:, value] = np.array((nt['center_x'][diff],
                                                                    nt['center_y'][diff],
                                                                    nt['xVelocity'][diff],
                                                                    nt['yVelocity'][diff],
                                                                    nt['xAcceleration'][diff],
                                                                    nt['yAcceleration'][diff])).transpose()
        fill_inp_v_start = num_agent_features + 24
        fill_inp_v_end = num_agent_features + 30
        inp_v[fill_inp_v_start:fill_inp_v_end] = leftPreceding_nbr_features


        # build rightFollowing_nbr feature table
        rightFollowing_nbr_features = np.zeros(
            (num_neighbor_features, inp_v.shape[1]))  # inp_v.shape[1] : length of trajectory/track of the agent
        unique_nbr = np.unique(rightFollowing_nbr)  # list of preceding neighbors throughout the track
        unique_nbr_indices = []  # list the indices at which the unique neighbors appears
        for i in range(len(unique_nbr)):
            unique_nbr_indices.append(np.where(rightFollowing_nbr == unique_nbr[i]))
            if unique_nbr[i] != 0:
                nt = tracks[unique_nbr[i] - 1]  # -1 since tracks serial number starts from 0
                idx = unique_nbr_indices[i]
                for j, value in enumerate(idx[0]):  # value is the index number of the agent where this nbr appears
                    frame_ref = t['frame'][value]  # what is the corresponding frame number for this index
                    nbr_start_frame = nt['frame'][0]
                    diff = frame_ref - nbr_start_frame  # reference frame
                    rightFollowing_nbr_features[:, value] = np.array((nt['center_x'][diff],
                                                                     nt['center_y'][diff],
                                                                     nt['xVelocity'][diff],
                                                                     nt['yVelocity'][diff],
                                                                     nt['xAcceleration'][diff],
                                                                     nt['yAcceleration'][diff])).transpose()
        fill_inp_v_start = num_agent_features + 30
        fill_inp_v_end = num_agent_features + 36
        inp_v[fill_inp_v_start:fill_inp_v_end] = rightFollowing_nbr_features

        # build rightAlongside_nbr feature table
        rightAlongside_nbr_features = np.zeros(
            (num_neighbor_features, inp_v.shape[1]))  # inp_v.shape[1] : length of trajectory/track of the agent
        unique_nbr = np.unique(rightAlongside_nbr)  # list of preceding neighbors throughout the track
        unique_nbr_indices = []  # list the indices at which the unique neighbors appears
        for i in range(len(unique_nbr)):
            unique_nbr_indices.append(np.where(rightAlongside_nbr == unique_nbr[i]))
            if unique_nbr[i] != 0:
                nt = tracks[unique_nbr[i] - 1]  # -1 since tracks serial number starts from 0
                idx = unique_nbr_indices[i]
                for j, value in enumerate(idx[0]):  # value is the index number of the agent where this nbr appears
                    frame_ref = t['frame'][value]  # what is the corresponding frame number for this index
                    nbr_start_frame = nt['frame'][0]
                    diff = frame_ref - nbr_start_frame  # reference frame
                    rightAlongside_nbr_features[:, value] = np.array((nt['center_x'][diff],
                                                                      nt['center_y'][diff],
                                                                      nt['xVelocity'][diff],
                                                                      nt['yVelocity'][diff],
                                                                      nt['xAcceleration'][diff],
                                                                      nt['yAcceleration'][diff])).transpose()
        fill_inp_v_start = num_agent_features + 36
        fill_inp_v_end = num_agent_features + 42
        inp_v[fill_inp_v_start:fill_inp_v_end] = rightAlongside_nbr_features


        # build rightPreceding_nbr feature table
        rightPreceding_nbr_features = np.zeros(
            (num_neighbor_features, inp_v.shape[1]))  # inp_v.shape[1] : length of trajectory/track of the agent
        unique_nbr = np.unique(rightPreceding_nbr)  # list of preceding neighbors throughout the track
        unique_nbr_indices = []  # list the indices at which the unique neighbors appears
        for i in range(len(unique_nbr)):
            unique_nbr_indices.append(np.where(rightPreceding_nbr == unique_nbr[i]))
            if unique_nbr[i] != 0:
                nt = tracks[unique_nbr[i] - 1]  # -1 since tracks serial number starts from 0
                idx = unique_nbr_indices[i]
                for j, value in enumerate(idx[0]):  # value is the index number of the agent where this nbr appears
                    frame_ref = t['frame'][value]  # what is the corresponding frame number for this index
                    nbr_start_frame = nt['frame'][0]
                    diff = frame_ref - nbr_start_frame  # reference frame
                    rightPreceding_nbr_features[:, value] = np.array((nt['center_x'][diff],
                                                                      nt['center_y'][diff],
                                                                      nt['xVelocity'][diff],
                                                                      nt['yVelocity'][diff],
                                                                      nt['xAcceleration'][diff],
                                                                      nt['yAcceleration'][diff])).transpose()
        fill_inp_v_start = num_agent_features + 42
        fill_inp_v_end = num_agent_features + 48
        inp_v[fill_inp_v_start:fill_inp_v_end] = rightPreceding_nbr_features

        inputs.append(inp_v)

    return inputs

#    print('done')
def get_data():

    for i in range(1,61):
        number =  "{0:2d}".format(i)
        track_path = "data/" + number + "_tracks.csv"
        track_meta_path = "data/"+ number + "_tracksMeta.csv"
    #path_01 = "E:/Research/Implementation/project_1/data/01_tracks.csv"
    #track_meta_path_01 = "E:/Research/Implementation/project_1/data/01_tracksMeta.csv"
    #tracks, tracks_meta = get_dataset(path_01, track_meta_path_01)
        tracks, tracks_meta = get_dataset(track_path, track_meta_path)

        inputs1 = get_input_vector(tracks)
    '''
    path_02= "E:/Research/Implementation/project_1/data/02_tracks.csv"
    tracks = get_dataset(path_02)
    inputs2 = get_input_vector(tracks)

    path_03= "E:/Research/Implementation/project_1/data/03_tracks.csv"
    tracks = get_dataset(path_03)
    inputs3 = get_input_vector(tracks)

    path_04= "E:/Research/Implementation/project_1/data/04_tracks.csv"
    tracks = get_dataset(path_04)
    inputs4 = get_input_vector(tracks)

    path_05= "E:/Research/Implementation/project_1/data/05_tracks.csv"
    tracks = get_dataset(path_05)
    inputs5 = get_input_vector(tracks)

    path_06= "E:/Research/Implementation/project_1/data/06_tracks.csv"
    tracks = get_dataset(path_06)
    inputs6 = get_input_vector(tracks)

    path_07= "E:/Research/Implementation/project_1/data/07_tracks.csv"
    tracks = get_dataset(path_07)
    inputs7 = get_input_vector(tracks)

    inputs = inputs1 + inputs2 + inputs3 + inputs4 + inputs5 + inputs6 + inputs7
    '''
    return inputs1


'''
from tempfile import TemporaryFile
input_data = TemporaryFile()
np.save(input_data, inputs)


print('done')
'''

#get_data()
def acceptable_length(trajectories):  # frame numbers of some trajectories are not high enough; remove the onws that has smaller number of frames
    obs_len = 50  # 50 frames  equivalent to 2 seconds
    pred_len = 75  # 75 frames, equivalent to 3 seconds
    for i in range (len(trajectories)):
        if len(trajectories[i][1]) < obs_len +pred_len:
            trajectories.remove(trajectories[i])
    return trajectories


#new_inputs = acceptable_length(inputs)

