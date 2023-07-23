This repository is an experiment to solve multimoda-single-cell-integration(https://www.kaggle.com/competitions/open-problems-multimodal) using pysaprk.
While most solutions end up using stand alone libraries like sklearn, using spark allows distributed processing hence this solution can be used over a much larger dataset than one given in the problem for further research.

File(s) in use 
1. incremental_pca.py - a naive implementation of incremental PCA (since I couldn't't find impl for this in spark, neither could I find impl for truncated SVD which is a more effecient dimentionality reduction process supported in sklearn) and reduces dimentaions of input features from ~(100k x 220k) to ~(100k x 58).
2. predict_gene_expression.py - given index of a test target, it generates prediction for that target by training on the PCA output. It takes one minute to generate prediction for one target, one gene in the context of the problem. This is not really helpful since there are ~23000 genes to be worked with. At the moment I'm unable to parallelise the process of training and predicting multiple target genesbecause of the limitations of the spark context.
