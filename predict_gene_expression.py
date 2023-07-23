import logging
from typing import Tuple, Dict

import h5py
import hdf5plugin
import numpy as np
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import Row

logging.info(f"hdf5plugin at ({hdf5plugin.PLUGIN_PATH}) is necessary for h5py to work")

def get_spark_session() -> SparkSession:
    return SparkSession.builder \
        .appName("multimodal-single-cell-integration") \
        .config("spark.executor.memory", "12g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.default.parallelism", 8) \
        .getOrCreate()


spark = get_spark_session()


def get_train_gene_expression_paths() -> Tuple[str, str, str, str]:
    train_gene_expression_h5_file_path = '/Users/niranjani/code/multimodal-single-cell-integration/data/raw_data/train_multi_targets.h5'
    train_gene_expression_matrix_path = 'train_multi_targets/block0_values'
    train_gene_ids_path = 'train_multi_targets/block0_items'
    # train_gene_ids_path = 'train_multi_targets/axis0'
    train_cell_ids_path = 'train_multi_targets/axis1'
    return train_gene_expression_h5_file_path, train_cell_ids_path, train_gene_ids_path, train_gene_expression_matrix_path


def get_test_features() -> DataFrame:
    def get_test_chromatin_accessibility_pca_df() -> DataFrame:
        test_chromatin_accessibility_pca_path = '/Users/niranjani/code/multimodal-single-cell-integration/data/pca/test_multiome_chromatin_accessibility/incremental_pca'
        return spark.read.parquet(test_chromatin_accessibility_pca_path)

    return get_test_chromatin_accessibility_pca_df()


def get_train_features() -> DataFrame:
    def get_train_chromatin_accessibility_pca_df() -> DataFrame:
        train_chromatin_accessibility_pca_path = '/Users/niranjani/code/multimodal-single-cell-integration/data/pca/multiome_chromatin_accessibility/incremental_pca'
        return spark.read.parquet(train_chromatin_accessibility_pca_path)

    return get_train_chromatin_accessibility_pca_df()


def get_training_df(training_features: DataFrame, metadata: Dict) -> DataFrame:
    def get_gene_expression_df_for_gene(metadata: Dict) -> DataFrame:
        train_gene_expression_h5_file_path, _, _, train_gene_expression_matrix_path = get_train_gene_expression_paths()
        file = h5py.File(train_gene_expression_h5_file_path, 'r')
        gene_expression_for_this_gene = np.array(
            file[train_gene_expression_matrix_path][:, metadata["gene_id_index"]])
        data = [Row(target=float(value), rowIndex=index) for index, value in
                enumerate(gene_expression_for_this_gene)]
        return spark.createDataFrame(data)

    def get_training_targets(metadata: Dict) -> DataFrame:
        return get_gene_expression_df_for_gene(metadata)

    join_key = 'rowIndex'
    training_targets = get_training_targets(metadata)
    training_df = training_features.join(training_targets, on=join_key, how="inner")
    return training_df


if __name__ == '__main__':
    lr = LinearRegression(featuresCol="features", labelCol="target")
    train_features = get_train_features()
    test_features = get_test_features()
    gene_ids_indices = [17302, 17303, 17304, 17305, 17306]
    for gene_id_index in gene_ids_indices:
        training_df = get_training_df(train_features, metadata={"gene_id_index": gene_id_index})
        model = lr.fit(training_df)
        predictions = model.transform(test_features)
        predictions.show()
