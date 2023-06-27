import random
from typing import List, Tuple

import h5py
import hdf5plugin
import numpy as np
from joblib import Parallel, delayed
from pyspark.sql import SparkSession
from pyspark.ml.linalg import SparseMatrix

SAMPLING_SIZE = 1000


def get_csc_values_for_matrix(matrix: np.ndarray) -> Tuple[int, int, List[int], List[int], List]:
    row_cnt, column_cnt = matrix.shape

    values = []
    row_indices = []
    column_pointers = [0]

    def process_column(j):
        non_zero_values = []
        non_zero_indices = []
        for i in range(row_cnt):
            if matrix[i, j] != 0:
                non_zero_values.append(matrix[i, j])
                non_zero_indices.append(i)
        return non_zero_values, non_zero_indices

    non_zero_values_with_indices_for_all_columns = Parallel(n_jobs=-1)(
        delayed(process_column)(j) for j in range(column_cnt))

    for j in range(column_cnt):
        non_zero_values, non_zero_indices = non_zero_values_with_indices_for_all_columns[j]
        values.extend(non_zero_values)
        row_indices.extend(non_zero_indices)
        column_pointers.append(column_pointers[-1] + len(non_zero_values))

    return row_cnt, column_cnt, column_pointers, row_indices, values


def get_spark_session() -> SparkSession:
    return SparkSession.builder \
        .appName("multimodal-single-cell-integration") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()


def write_subset_as_sparse_matrix_dataframe(
        h5_file_path: str,
        row_index_path: str,
        matrix_path: str,
        output_path: str,
):
    file = h5py.File(h5_file_path, 'r')
    sampled_row_index = random.sample(range(0, file[row_index_path].shape[0]), SAMPLING_SIZE)
    sampled_matrix = np.stack(
        tuple([file[matrix_path][i, :] for i in sampled_row_index])
    )

    row_cnt, column_cnt, column_pointers, row_indices, values = get_csc_values_for_matrix(sampled_matrix)

    values = np.array(values)
    row_indices = np.array(row_indices)
    column_pointers = np.array(column_pointers)

    sparse_matrix = SparseMatrix(row_cnt, column_cnt, column_pointers, row_indices, values)

    data = [(sparse_matrix.numRows, sparse_matrix.numCols, sparse_matrix.colPtrs.tolist(), sparse_matrix.rowIndices.tolist(),
             sparse_matrix.values.tolist())]
    get_spark_session().createDataFrame(
        data, ["num_rows", "num_cols", "col_ptrs", "row_indices", "values"]
    ).write.parquet(output_path)


train_multi_inputs_path = '/Users/niranjani/code/multimodal-single-cell-integration/data/raw_data/train_multi_inputs.h5'
cell_ids_path = 'train_multi_inputs/axis1'
genomic_coordinates_path = 'train_multi_inputs/block0_items'
chromatin_accessibility_matrix_path = 'train_multi_inputs/block0_values'
chromatin_accessibility_of_sample_cells_path = 'data/subset'

write_subset_as_sparse_matrix_dataframe(
    train_multi_inputs_path,
    cell_ids_path,
    chromatin_accessibility_matrix_path,
    chromatin_accessibility_of_sample_cells_path,
)
