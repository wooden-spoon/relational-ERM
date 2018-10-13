"""
Load and process pokec social network

Data includes:
edge structure
user profiles

data from https://snap.stanford.edu/data/soc-Pokec.html
"""

import os
import numpy as np
import pandas as pd
import gzip

def _load_profiles(profile_file):
    # df = pd.read_csv('filename.tar.gz', compression='gzip', header=0, sep=',', quotechar='"'

    names = str.split("user_id public completion_percentage gender region last_login registration age body "
                      "I_am_working_in_field spoken_languages hobbies I_most_enjoy_good_food pets body_type "
                      "my_eyesight eye_color hair_color hair_type completed_level_of_education favourite_color "
                      "relation_to_smoking relation_to_alcohol sign_in_zodiac on_pokec_i_am_looking_for love_is_for_me "
                      "relation_to_casual_sex my_partner_should_be marital_status children relation_to_children "
                      "I_like_movies I_like_watching_movie I_like_music I_mostly_like_listening_to_music "
                      "the_idea_of_good_evening I_like_specialties_from_kitchen fun I_am_going_to_concerts "
                      "my_active_sports my_passive_sports profession I_like_books life_style music cars politics "
                      "relationships art_culture hobbies_interests science_technologies computers_internet education "
                      "sport movies travelling health companies_brands more")

    usecols = str.split("user_id public completion_percentage gender region last_login registration age")

    profiles = pd.read_csv(profile_file, names=names, index_col=False, usecols=usecols, compression='gzip', header=None, sep='\t')
    profiles.set_index('user_id', inplace=True, drop=False)
    return profiles


def _process_profiles(profiles):
    """
    Subset the profiles to strip out attributes that are freely fillable by users
    Fix datatypes for remainders
    """
    # keep_attributes = str.split("user_id public completion_percentage gender region last_login registration age")
    # p2=profiles[keep_attributes]
    p2 = profiles
    p2['region'] = p2['region'].astype('category')
    p2['public'] = p2['public'].astype('category')
    p2['gender'] = p2['gender'].astype('category')
    p2['last_login'] = pd.to_datetime(p2['last_login'])
    p2['registration'] = pd.to_datetime(p2['registration'])
    p2.loc[p2.age == 0, 'age'] = np.nan

    return p2


def preprocess_data(data_directory='../data/pokec'):
    link_file = os.path.join(data_directory, 'soc-pokec-relationships.txt.gz')
    profile_file = os.path.join(data_directory, 'soc-pokec-profiles.txt.gz')

    edge_list = np.loadtxt(link_file, dtype=np.int32)
    profiles = _load_profiles(profile_file)
    profiles = _process_profiles(profiles)

    # relational ERM code expects 0-indexed vertices, but data is 1-indexed
    profiles['user_id'] = profiles['user_id'] - 1
    edge_list = edge_list - 1

    return {
        'edge_list': edge_list,
        'profiles': profiles
    }


def preprocess_packed_adjacency_list(data):
    from ..graph_ops.representations import create_packed_adjacency_from_edge_list

    # Load the current edge list, and go to canonical form
    # i.e., remove self-edges and convert to undirected graph
    edge_list = data['edge_list']
    edge_list = edge_list[edge_list[:, 0] != edge_list[:, 1], :]
    edge_list.sort(axis=-1)
    edge_list = np.unique(edge_list, axis=0)

    # Compute redundant edge list
    edge_list = np.concatenate((edge_list, np.flip(edge_list, axis=1)))
    packed_adj_list = create_packed_adjacency_from_edge_list(edge_list)

    return {
        'neighbours': packed_adj_list.neighbours,
        'lengths': packed_adj_list.lengths,
        # 'sparse_labels': data['labels']
    }


def main():
    data = preprocess_data()
    data['profiles'].to_pickle("profiles.pkl")
    np.savez_compressed('pokec_links.npz', {'edge_list' : data['edge_list']})

    link_data = preprocess_packed_adjacency_list(data)
    np.savez_compressed('pokec_links_processed.npz', **link_data)


if __name__ == '__main__':
    main()

