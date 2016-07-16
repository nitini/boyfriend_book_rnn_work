import json
import pandas as pd
import file_paths as fp
import sys
import numpy as np
import time

#%%
def extract_stanzas(file_name):
    hit_pre = 0
    hit_pre_end = 0
    first_pre = 0
    stanzas = []
    current_stanza = ''
    line_count = 0
    with open(file_name, 'rb') as f:
        for line in f:

            if line_count == 0 and 'artist' in line.lower():
                hit_pre = 1
                first_pre = 1
                print(file_name)

            line_count += 1

            if '<pre>' in line:
                hit_pre = 1
                first_pre = 1

            if '</pre>' in line:
                hit_pre_end = 1

            if hit_pre_end == 1:
                break

            if hit_pre == 1:
                if first_pre == 1:
                    first_pre = 0
                    continue
                formatted_line = line.decode('ascii', 'ignore').encode('ascii', 'ignore').lower()
                if formatted_line.strip(' ') == '\n':
                    stanzas.append(current_stanza)
                    current_stanza = ''
                else:
                    current_stanza += formatted_line
    artist = stanzas[0].split('\n')[0].split(': ')[-1]
    album = stanzas[0].split('\n')[1].split(': ')[-1]
    song = stanzas[0].split('\n')[2].split(': ')[-1]

    song_stanza_data = pd.DataFrame(stanzas, columns=['stanza'], index=range(len(stanzas)))

    song_stanza_data['len_of_stanza'] = song_stanza_data.stanza.apply(lambda x: len(x))
    song_stanza_data['artist'] = artist.strip()
    song_stanza_data['song_title'] = song.strip()
    song_stanza_data['album'] = album.strip()
    song_stanza_data['file_name'] = file_name.split('/')[-1]

    return song_stanza_data.sort_values(by='len_of_stanza', ascending=False)


def compile_stanzas(lyric_folders):
    stanzas = pd.DataFrame(columns=['stanza','len_of_stanza','artist','song_title','album'])
    for folder in lyric_folders:
        for file in os.listdir(folder):
            print(file)
            if file[-3:] != 'txt':
                continue
            stanza_data = extract_stanzas(folder + file)
            stanzas = pd.concat([stanzas, stanza_data],ignore_index=True)

    return stanzas



def extract_songs(song_folder):
    song_files = os.listdir(song_folder)
    for song_file in song_files:
        lyrics = strip_html(song_folder + song_file)

    return



