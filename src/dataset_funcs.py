import json
from itertools import islice
import re
import os
import collections
from settings import mainDataSetPath
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


# def get_track_song_data_set():
#    return get_playlist_data_set('track_name')
# def get_track_album_data_set():
#    return get_playlist_data_set('album_name')
# def get_track_artist_data_set():
#    return get_playlist_data_set('artist_name')


def append_result(output):

    print("LOG: Appending Result to result_output.txt")
    # text_file = open(outputFile, "w")
    # output = str(output).replace(', (', '\n')
    # output = str(output).replace(')', '')
    # output = str(output).replace('\'', '')
    # output = str(output).replace('[(', '')
    # text_file.write(str(output))
    # text_file.close()


def str_cleansing(str_input):
    # str_input = str_input.lower()
    # str_input = re.sub(r"[.,\/#!$%\^\*;:{}=\_`~()@]", ' ', str_input)
    # str_input = re.sub(r'\s+', ' ', str_input).strip()
    # return str(str_input)
    pass




def levenshtein_distance(a, b):
    return #fuzz.token_sort_ratio(a, b)
    # print("LOG: similarity - " + str(similarity))


"""
import multiprocessing
entry_queue = multiprocessing.Queue()
def multiprocess_parse():
    global i
    datasetlist = os.listdir(mainDataSetPath)
    track_data_set = "'"
    i = 0
    for ds in sorted(datasetlist):
        i += 1
        print("LOG: Analyzing data_set [" + str(i) + " / 1000]")
        if ds.startswith("mpd.slice.") and ds.endswith(".json") and i<100:
            fullpath = os.sep.join((mainDataSetPath, ds))
            with open(fullpath, 'r') as json_file:
                # js = json_file.read()
                data_set = json.load(json_file)
                for p in data_set['playlists']:
                    try:
                        # for t in p['tracks']:
                        #     track_data_set += str_cleansing(t[search_input]) + ','
                        entry_queue.put(p)
                    except KeyError:
                        pass
    # return track_data_set[:-1] + "'"
    show_top25()
def mapreduce_process(queue):
    global total_playlists, total_tracks, TOP25_TITLES, TOP25_ARTISTS, TOP25_SONGS, TOP25_ALBUMS

    print(os.getpid(), "working")
    while True:
        item = queue.get(True)
        total_playlists += 1
        TOP25_TITLES[item['name'].lower()] += 1
        for track in item['tracks']:
            total_tracks += 1
            TOP25_ARTISTS[str_cleansing(track['artist_name'])] += 1
            TOP25_SONGS[str_cleansing(track['track_name'])] += 1
            TOP25_ALBUMS[str_cleansing(track['album_name'])] += 1
the_pool = multiprocessing.Pool(4, mapreduce_process, (entry_queue,))
"""
"""
import json
from itertools import islice
import re
import os
import collections
from settings import mainDataSetPath
from src.dataset_funcs import str_cleansing, levenshtein_distance
from src.core.entities import Core as core, titleHashMap
count = 0

trainTitleList = []


def add_index(playlist, pid):
    # if keyword exists add otherwise make new entry
    # titleHashMap.setdefault(playlist, []).append(pid)
    pass

def article_to_dict():
    # global i
    # i = 0
    # print("LOG: Analyzing challenge data_set")
    #
    # with open(challengeDataSet, 'r') as json_file:
    #     data_set = json.load(json_file)
    #     for p in data_set['playlists']:
    #         try:
    #             add_index(str_cleansing(p['name']), p['pid'])
    #         except KeyError:
    #             pass
    # # core.add_ptitles(None) # ADD TO DATABASE
    # create_train_list()
    # match_playlists()
    return


def match_playlists():
    pass
    # global trainTitleList
    # for ch_name, v in titleHashMap.items():
    #     print(str(ch_name))
    #     for name, pid in trainTitleList:
    #         try:
    #             if levenshtein_distance(name, ch_name) == 100:
    #                 print("{ " + str(ch_name) + " } - { " + str(name) + " }")
    #         except KeyError:
    #             pass

"""