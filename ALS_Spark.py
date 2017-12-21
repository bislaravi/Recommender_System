from pyspark.sql import SparkSession, functions, types
spark = SparkSession.builder.appName('example application').getOrCreate()
sc = spark.sparkContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import pandas as pd



trainfile = 'train_rating.txt'
testfile = 'test_rating.txt'

# trainfile = '/home/ravibisla/PycharmProjects/DataScience/train_rating.txt'
# testfile = '/home/ravibisla/PycharmProjects/DataScience/test_rating.txt'

trainDf=spark.read.csv(trainfile,header='true')
testDf=spark.read.csv(testfile,header='true')
# trainDf1=trainDf.select('user_id','business_id','rating')
# testDf1=testDf.select('user_id','business_id','rating')

trainDf2 = trainDf.select(trainDf.user_id.cast('int').alias('userid'),trainDf.business_id.cast('int').alias('business_id'),trainDf.rating.cast('float').alias('rating'))
testDf2 = testDf.select(testDf.user_id.cast('int').alias('userid'),testDf.business_id.cast('int').alias('business_id'))


(training, test) = trainDf2.randomSplit([0.8, 0.2])
# Build the recommendation model using ALS on the training data


# als = ALS(maxIter=20, regParam=0.1,rank=20, userCol="userid", itemCol="business_id", ratingCol="rating",
#           coldStartStrategy="drop")
# als = ALS(maxIter=20, regParam=0.1,rank=20, userCol="userid", itemCol="business_id", ratingCol="rating",
#           coldStartStrategy="drop")  # 1.6653266320794866
als = ALS(maxIter=20, regParam=0.1,rank=20, userCol="userid", itemCol="business_id", ratingCol="rating",
          coldStartStrategy="drop")  # 1.6653266320794866
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data


predictions = model.transform(test)
#Masking

predpand=predictions.toPandas()
mask = predpand.prediction > 5
mask1 = predpand.prediction < 0
column_name = 'prediction'
predpand.loc[mask, column_name] = 5
predpand.loc[mask1, column_name] = 1
evall= spark.createDataFrame(predpand)

# End Masking
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")

rmse = evaluator.evaluate(evall)
print("Root-mean-square error = " + str(rmse))