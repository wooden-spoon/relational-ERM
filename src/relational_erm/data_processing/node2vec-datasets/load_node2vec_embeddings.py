"""
Load and process embeddings output by reference implementation of node2vec
"""

import os
import numpy as np
import pandas as pd


def process_n2v_emb(data_dir, emb_file):
    n2v_emb_file = os.path.join(data_dir, emb_file)

    hs_embeddings=pd.read_csv(n2v_emb_file, delim_whitespace=True, skiprows=1, index_col=0, header=None)

    tmp = hs_embeddings.as_matrix()
    hs_mat=np.empty_like(tmp)
    hs_mat[hs_embeddings.index] = tmp

    return hs_mat

def main():
    data_dir = '../data/homo_sapiens'
    # emb_file = 'homo_sapiens.emd'
    # save_file = 'homo_sapiens_embed.npz'
    emb_file = 'node2vec_emb.emd'
    save_file = 'node2vec_emb.npz'
    np_embeddings = process_n2v_emb(data_dir, emb_file)

    data_dir = '../data/blog_catalog_3'
    # emb_file = 'blog_catalog.emd'
    save_file = 'blog_catalog_erm_embed.npz'
    # emb_file = 'blog_catalog_nodownsamp.emd'
    # save_file = 'blog_catalog_nodownsamp_embed.npz'
    # emb_file = 'node2vec_emb.emd'
    # save_file = 'node2vec_emb.npz'

    np_embeddings = process_n2v_emb(data_dir, emb_file)
    save_path = os.path.join(data_dir, save_file)
    np.savez_compressed(save_path, embeddings=np_embeddings)


if __name__ == '__main__':
    main()