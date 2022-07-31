package com.snowave.machine.learning.KMeans

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

/**
 * 通过数据集使用kmeans训练模型
 *
 * 0.0 0.0 0.0
 * 0.1 0.1 0.1
 * 0.2 0.2 0.2
 * 9.0 9.0 9.0
 * 9.1 9.1 9.1
 * 9.2 9.2 9.2
 */
object KMeansScala {
  def main(args: Array[String]): Unit = {

    //1 构建Spark对象
    val conf = new SparkConf().setAppName("KMeans").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setLogLevel("Error")

    // 读取样本数据1，格式为LIBSVM format
    val data = sc.textFile("./sample-data/kmeans_data.txt")
    // 由于没有y值, 所以直接定义vector, victors就是features
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    val numClusters = 2 // 设置两个中心点
    val numIterations = 100
    val model = new KMeans().
      //设置聚类的类数
      setK(numClusters).
      //设置找中心点最大的迭代次数
      setMaxIterations(numIterations).run(parsedData)

    //2个中心点的坐标
    val centers = model.clusterCenters
    val k = model.k
    println("中心点坐标：")
    centers.foreach(println)
    println("类别 k = " + k)
    //保存模型
    model.save(sc, "./Kmeans_model")
    //加载模型
    val sameModel = KMeansModel.load(sc, "./Kmeans_model")

    // 展示样本分类结果, 预测一下点(1,1,1)的结果属于哪一类
    println("测试样本分类号(1,1,1)：" + sameModel.predict(Vectors.dense(1, 1, 1)))

    //SparkSQL读取显示2个中心点坐标
    val spark = SparkSession.builder().getOrCreate()
    spark.read.parquet("./Kmeans_model/data").show() // 读取保存好的模型
  }
}