import random

import h5py
import hdf5plugin
import numpy as np
import pandas as pd

SAMPLING_SIZE = 1000

cell_ids_path = 'train_multi_inputs/axis1'
genomic_coordinates_path = 'train_multi_inputs/block0_items'
chromatin_accessibility_matrix_path = 'train_multi_inputs/block0_values'
train_multi_inputs_path = '/Users/niranjani/code/multimodal-single-cell-integration/data/raw_data/train_multi_inputs.h5'
chromatin_accessibility_of_sample_cells_path = '/Users/niranjani/code/multimodal-single-cell-integration/data/subset/chromatin_accessibility.parquet'

train_multi_inputs = h5py.File(train_multi_inputs_path, 'r')

random_cell_id_sample_index = random.sample(range(0, train_multi_inputs[cell_ids_path].shape[0]), SAMPLING_SIZE)
chromatin_accessibility_of_sample_cells = np.stack(
    tuple([train_multi_inputs[chromatin_accessibility_matrix_path][i, :] for i in random_cell_id_sample_index]))
genomic_coordinates = train_multi_inputs[genomic_coordinates_path][:]
genomic_coordinates_index = [str(num) for num in range(genomic_coordinates.shape[0])]

df = pd.DataFrame(chromatin_accessibility_of_sample_cells, columns=genomic_coordinates_index)
df.to_parquet(chromatin_accessibility_of_sample_cells_path)
