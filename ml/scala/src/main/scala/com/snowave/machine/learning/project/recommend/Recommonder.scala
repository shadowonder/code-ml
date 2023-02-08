package com.snowave.machine.learning.project.recommend

import java.io.PrintWriter

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.Map

/**
 * 对 load 到本地的数据集 进行处理
 * 主要是对这批数据集 构建全部特征维度 
 */
object Recommonder {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("recommonder").setMaster("local[*]")
    val sc = new SparkContext(conf)
    //加载数据，用\t分隔开  -- 1	Item.id,hitop_id35:1;Item.screen,screen3:1;Item.name,ch_name54:1
    val data: RDD[Array[String]] = sc.textFile("./traindata", 8).map(_.split("\t"))

    //得到第一列的值，也就是label
    val label: RDD[String] = data.map(_ (0))

    //sample这个RDD中保存的是每一条记录的特征名
    //-1	Item.id,hitop_id53:1;Item.screen,screen6:1;Item.name,ch_name80:1;Item.author,author1:1
    val sample: RDD[Array[String]] = data.map(_ (1)).map(x => {
      val arr: Array[String] = x.split(";").map(_.split(":")(0))
      arr
    })
    //将所有元素压平，得到的是所有分特征，然后去重，最后索引化，也就是加上下标，最后转成map是为了后面查询用
    //dict 是所有数据的所有不重复的特征
    val allFeaturesMap: Map[String, Long] = sample.flatMap(x => x).distinct().zipWithIndex().collectAsMap()
    //得到稀疏向量，为每条数据的features，与dict对比，缺少的特征补成0
    val sam: RDD[SparseVector] = sample.map(sampleFeaturesArray => {
      //index中保存的是，未来在构建训练集时，下面填1的索引号集合
      val indexArray: Array[Int] = sampleFeaturesArray.map(feature => {
        //get出来的元素程序认定可能为空，做一个类型匹配
        val currentFeaturesIndex: Long = allFeaturesMap(feature)
        //非零元素下标，转int符合SparseVector的构造函数
        currentFeaturesIndex.toInt
      })
      //SparseVector创建一个向量
      new SparseVector(allFeaturesMap.size, indexArray, Array.fill(indexArray.length)(1.0))
    })

    //mllib中的逻辑回归只认1.0和0.0，这里进行一个匹配转换
    val trainData: RDD[LabeledPoint] = label.map {
      case "-1" => 0.0
      case "1" => 1.0
    }.zip(sam).map(x => new LabeledPoint(x._1, x._2))

    //逻辑回归训练，两个参数，迭代次数和步长，生产常用调整参数
    val lr = new LogisticRegressionWithSGD()
    // 设置W0截距
    lr.setIntercept(true)

    //    // 设置正则化
    //    lr.optimizer.setUpdater(new SquaredL2Updater)
    //    // 看中W模型推广能力的权重
    //    lr.optimizer.setRegParam(0.4)

    // 最大迭代次数
    lr.optimizer.setNumIterations(100)
    // 设置梯度下降的步长,学习率
    lr.optimizer.setStepSize(0.01)

    val model: LogisticRegressionModel = lr.run(trainData)

    //模型结果权重
    val weights: Array[Double] = model.weights.toArray
    //将map反转，weights相应下标的权重对应map里面相应下标的特征名
    val swapMap: Map[Long, String] = allFeaturesMap.map(_.swap)
    //模型保存
    //    LogisticRegressionModel.load()
    //    model.save()
    //输出
    val pw = new PrintWriter("./model");
    //遍历
    for (i <- weights.indices) {
      //通过map得到每个下标相应的特征名
      val featureName = swapMap(i)
      //      val featureName = swapMap.get(i)match {
      //        case Some(feature) => feature
      //        case None => ""
      //      }
      //特征名对应相应的权重
      val str = featureName + "\t" + weights(i)
      pw.write(str)
      pw.println()
    }
    pw.flush()
    pw.close()
  }
}