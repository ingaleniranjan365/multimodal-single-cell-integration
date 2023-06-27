import random
from typing import List, Tuple

import h5py
import hdf5plugin
import numpy as np
from joblib import Parallel, delayed
from pyspark.sql import SparkSession
from pyspark.ml.linalg import SparseMatrix


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
        column_index_path: str,
        matrix_path: str,
        output_path: str,
        row_sampling_size: int,
        column_sampling_size: int,
        row_label: str,
        column_label: str,
        matrix_label: str,
        dataset_label: str,
):
    file = h5py.File(h5_file_path, 'r')

    sampled_row_values, sampled_column_values, sampled_matrix = get_sampled_dataset(file, matrix_path, row_index_path,
                                                                                    column_index_path,
                                                                                    row_sampling_size,
                                                                                    column_sampling_size)

    row_cnt, column_cnt, column_pointers, row_indices, values = get_csc_values_for_matrix(sampled_matrix)

    values = np.array(values)
    row_indices = np.array(row_indices)
    column_pointers = np.array(column_pointers)

    sparse_matrix = SparseMatrix(row_cnt, column_cnt, column_pointers, row_indices, values)

    matrix_data = [(sparse_matrix.numRows, sparse_matrix.numCols, sparse_matrix.colPtrs.tolist(),
                    sparse_matrix.rowIndices.tolist(),
                    sparse_matrix.values.tolist())]

    spark = get_spark_session()
    spark.createDataFrame(
        matrix_data, schema=["num_rows", "num_cols", "col_ptrs", "row_indices", "values"]
    ).write.parquet(f"{output_path}/{dataset_label}/sparse_matrix({matrix_label})")
    spark.createDataFrame([(sampled_row_values.tolist(),)], schema=[row_label]).write.parquet(
        f"{output_path}/{dataset_label}/rows({row_label})")
    spark.createDataFrame([(sampled_column_values.tolist(),)], schema=[column_label]).write.parquet(
        f"{output_path}/{dataset_label}/columns({column_label})")


def get_sampled_dataset(
        file: h5py.File,
        matrix_path: str,
        row_index_path: str,
        column_index_path: str,
        row_sample_size: int,
        column_sample_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_row_index = sorted(random.sample(range(0, file[row_index_path].shape[0]), row_sample_size))
    sampled_column_index = sorted(random.sample(range(0, file[column_index_path].shape[0]), column_sample_size))

    sampled_matrix = np.stack(
        tuple([np.stack(
            tuple([file[matrix_path][i, j] for j in sampled_column_index])
        ) for i in sampled_row_index])
    )
    sampled_rows = file[row_index_path][sampled_row_index]
    sampled_columns = file[column_index_path][sampled_column_index]
    return sampled_rows, sampled_columns, sampled_matrix


def get_row_sampled_matrix(file: h5py.File, matrix_path: str, row_index_path: str, row_sample_size: int):
    sampled_row_index = random.sample(range(0, file[row_index_path].shape[0]), row_sample_size)
    sampled_matrix = np.stack(
        tuple([file[matrix_path][i, :] for i in sampled_row_index])
    )
    return sampled_matrix


cell_id_sampling_size = 100
genomic_coordinates_sampling_size = 200
train_multi_inputs_path = '/Users/niranjani/code/multimodal-single-cell-integration/data/raw_data/train_multi_inputs.h5'
cell_ids_path = 'train_multi_inputs/axis1'
genomic_coordinates_path = 'train_multi_inputs/block0_items'
chromatin_accessibility_matrix_path = 'train_multi_inputs/block0_values'
chromatin_accessibility_of_sample_cells_path = 'data/subset'

write_subset_as_sparse_matrix_dataframe(
    train_multi_inputs_path,
    cell_ids_path,
    genomic_coordinates_path,
    chromatin_accessibility_matrix_path,
    chromatin_accessibility_of_sample_cells_path,
    cell_id_sampling_size,
    genomic_coordinates_sampling_size,
    "cell_ids",
    "genomic_coordinates",
    "multiome_chromatin_accessibility",
    "chromatin_accessibility"
)
