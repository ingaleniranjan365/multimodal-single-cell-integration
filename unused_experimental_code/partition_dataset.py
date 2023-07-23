import logging

import h5py
import hdf5plugin
import numpy as np
from joblib import Parallel, delayed
from pyspark.ml.linalg import SparseVector
from pyspark.sql import SparkSession

logger = logging.getLogger("multimodal-single-cell-integration")


def get_spark_session() -> SparkSession:
    return SparkSession.builder \
        .appName("multimodal-single-cell-integration") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

def partition_dataset(
        spark: SparkSession,
        batch_size: int,
        h5_file_path: str,
        matrix_path: str,
        output_path: str,
        matrix_label: str,
        dataset_label: str,
):
    file = h5py.File(h5_file_path, 'r')
    num_rows = file[matrix_path].shape[0]

    def process_row(index: int, row: np.ndarray):
        sparse_vector = SparseVector(len(row), [i for i, v in enumerate(row) if v != 0], [v for v in row if v != 0])
        return index, sparse_vector

    for i in range(0, num_rows, batch_size):
        start_index = i
        end_index = min(i + batch_size, num_rows)
        batch = file[matrix_path][start_index:end_index, :]

        logger.info(f"Processing rows {start_index} - {end_index}")

        sparse_matrix_with_index = Parallel(n_jobs=-1)(
            delayed(process_row)(i, row) for i, row in enumerate(batch)
        )
        sparse_matrix_with_index.sort(key=lambda x: x[0])

        df = spark.createDataFrame(sparse_matrix_with_index, ["dataset_row_index", "features"])
        df.write.parquet(
            f"{output_path}/subset_{start_index + 1}/{dataset_label}/sparse_matrix({matrix_label})")

        logger.info(
            f"generate output file - {output_path}/subset_{start_index + 1}/{dataset_label}/sparse_matrix({matrix_label})")


if __name__ == '__main__':
    spark = get_spark_session()

    output_path = '../data/subsets'
    batch_size = 10
    train_multi_inputs_path = '/data/raw_data/train_multi_inputs.h5'
    chromatin_accessibility_matrix_path = 'train_multi_inputs/block0_values'

    partition_dataset(
        spark,
        batch_size,
        train_multi_inputs_path,
        chromatin_accessibility_matrix_path,
        output_path,
        "multiome_chromatin_accessibility",
        "chromatin_accessibility"
    )
