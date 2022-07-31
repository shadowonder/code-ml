package com.ml.traffic

import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.{SparkConf, SparkContext}

object test {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("test")
    val sc = new SparkContext(conf)
    val model: LogisticRegressionModel = LogisticRegressionModel.load(sc, "hdfs://mynode1:8020/model/model_310999003001_1593499055278")
    val weights = model.weights
    val intercept = model.intercept
    println(weights)
    println(intercept)

  }

}
