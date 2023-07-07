import logging
from typing import Tuple, List

import h5py
import numpy as np
import hdf5plugin
from joblib import Parallel, delayed
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import VectorUDT
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf

logger = logging.getLogger("multimodal-single-cell-integration-scratch")
from pyspark.sql.functions import col


def get_spark_session() -> SparkSession:
    return SparkSession.builder \
        .appName("multimodal-single-cell-integration") \
        .config("spark.executor.memory", "12g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.default.parallelism", 12) \
        .getOrCreate()


def write_partial_pcas(
        spark: SparkSession,
        batch_size: int,
        h5_file_path: str,
        matrix_path: str,
        output_path: str
):
    def process_row(index: int, row: np.ndarray) -> Tuple[int, SparseVector]:
        non_zero_indices = np.nonzero(row)[0]
        non_zero_values = row[non_zero_indices]
        sparse_dict = {index: value for index, value in zip(non_zero_indices, non_zero_values)}
        sparse_vector = SparseVector(len(row), sparse_dict)
        return index, sparse_vector

    concat_vectors_udf = udf(
        lambda features, pca_features: Vectors.dense(np.concatenate((features.toArray(), pca_features.toArray()))),
        returnType=VectorUDT())

    partial_pca_paths = []

    num_partitions = spark.sparkContext.defaultParallelism
    file = h5py.File(h5_file_path, 'r')
    num_cols = file[matrix_path].shape[1]

    batches = [(i, i * batch_size) for i, _ in enumerate(range(0, num_cols, batch_size))]
    for batch in batches:
        batch_index, start_index = batch
        end_index = min(start_index + batch_size, num_cols)

        logging.info(f"starting PCA processing for columns {start_index}-{end_index}")

        batch = file[matrix_path][:, start_index:end_index]
        sparse_matrix_with_index = Parallel(n_jobs=-1)(
            delayed(process_row)(index, row) for index, row in enumerate(batch)
        )
        df = spark.createDataFrame(sparse_matrix_with_index, ["rowIndex", "features"])
        df = df.repartition(num_partitions)
        pca = PCA(k=1, inputCol="features", outputCol="pcaFeatures")
        model = pca.fit(df)
        batch_pca = model.transform(df)
        batch_pca = batch_pca.select("rowIndex", "pcaFeatures")
        partial_pca_path = f"{output_path}/{batch_index + 1}"
        batch_pca.repartition(1).write.parquet(partial_pca_path)
        partial_pca_paths.append(partial_pca_path)

    incremental_pca = spark.read.parquet(partial_pca_paths[0]).select(col("rowIndex"), col("pcaFeatures").alias("features"))
    for path in partial_pca_paths[1:]:
        batch_pca = spark.read.parquet(path)
        incremental_pca = incremental_pca.join(batch_pca, on="rowIndex")
        incremental_pca = incremental_pca.withColumn("incrementalFeatures",
                                                     concat_vectors_udf(incremental_pca["features"],
                                                                        incremental_pca["pcaFeatures"]))
        incremental_pca = incremental_pca.select(col("rowIndex"), col("incrementalFeatures").alias("features"))

    incremental_pca.write.parquet(f"{output_path}/incremental_pca")


if __name__ == '__main__':
    spark = get_spark_session()

    output_path = 'data/subsets/pca/multiome_chromatin_accessibility'
    batch_size = 4000

    train_multi_inputs_path = '/Users/niranjani/code/multimodal-single-cell-integration/data/raw_data/train_multi_inputs.h5'
    chromatin_accessibility_matrix_path = 'train_multi_inputs/block0_values'

    write_partial_pcas(
        spark,
        batch_size,
        train_multi_inputs_path,
        chromatin_accessibility_matrix_path,
        output_path
    )

    spark.stop()
