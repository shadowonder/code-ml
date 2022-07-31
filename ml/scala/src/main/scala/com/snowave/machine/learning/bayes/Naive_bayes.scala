package com.snowave.machine.learning.bayes

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


object Naive_bayes {
  def main(args: Array[String]) {
    //1 构建Spark对象
    val conf = new SparkConf().setAppName("Naive_bayes").setMaster("local")
    val sc = new SparkContext(conf)

    // 读取样本数据1
    // 0,1 0 0
    // 1,0 1 0
    // ...
    val data = sc.textFile("./sample-data/sample_naive_bayes_data.txt")
    val parsedData: RDD[LabeledPoint] = data.map { line =>
      val parts = line.split(',')
      // 逗号切割, 然后空格切割, 切割后转换为double
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }

    //样本数据划分训练样本与测试样本
    val splits = parsedData.randomSplit(Array(0.5, 0.5), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    //新建贝叶斯分类模型模型，并训练 ,lambda 拉普拉斯估计
    val model = NaiveBayes.train(training, lambda = 1.0)

    //对测试样本进行测试, 返回预测结果
    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))

    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    println(s"accuracy = $accuracy")
    val result = model.predict(Vectors.dense(Array[Double](0, 0, 100)))
    println("result = " + result)
    //保存模型
    //    val ModelPath = "./naive_bayes_model"
    //    model.save(sc, ModelPath)
    //    val sameModel = NaiveBayesModel.load(sc, ModelPath)
  }
}
