package com.ml.traffic

import java.text.SimpleDateFormat
import java.util
import java.util.Date

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

/**
 * 训练模型
 */
object TrainLRwithLBFGS {

  val sparkConf = new SparkConf().setAppName("train traffic model").setMaster("local[*]")
  val sc = new SparkContext(sparkConf)

  // create the date/time formatters
  val dayFormat = new SimpleDateFormat("yyyyMMdd")
  val minuteFormat = new SimpleDateFormat("HHmm")

  def main(args: Array[String]) {
    // fetch data from redis
    val jedis = RedisClient.pool.getResource
    jedis.select(1)
    // find relative road monitors for specified road
    // val camera_ids = List("310999003001","310999003102","310999000106","310999000205","310999007204")
    val camera_ids = List[String]("310999003001", "310999003102") // 对两条道论进行训练, 训练两个模型
    val camera_relations: Map[String, Array[String]] = Map[String, Array[String]](
      // 3001 对应的附近道路
      "310999003001" -> Array("310999003001", "310999003102", "310999000106", "310999000205", "310999007204"),
      // 3002 对应的道路
      "310999003102" -> Array("310999003001", "310999003102", "310999000106", "310999000205", "310999007204")
    )

    // 循环2个道路
    val temp = camera_ids.map({ camera_id =>
      val hours = 5
      val nowtimelong = System.currentTimeMillis()
      val now = new Date(nowtimelong)
      val day = dayFormat.format(now) //yyyyMMdd 格式化
      val array: Array[String] = camera_relations.get(camera_id).get

      /**
       * relations中存储了每一个卡扣在day这一天每一分钟的平均速度
       *
       * 遍历周围对应的道路
       */
      val relations: Array[(String, util.Map[String, String])] = array.map({ camera_id =>
        // fetch records of one camera for three hours ago  -- hgetAll(20200630_'310999003001')
        val minute_speed_car_map: util.Map[String, String] = jedis.hgetAll(day + "_'" + camera_id + "'")
        (camera_id, minute_speed_car_map)
      })

      // organize above records per minute to train data set format (MLUtils.loadLibSVMFile)
      // val dataSet = ArrayBuffer[LabeledPoint]() // arraybuffer 不能被序列化
      val dataSet = Array[LabeledPoint]()

      // start begin at index 3
      //Range 从300到1 递减 不包含0
      for (i <- Range(60 * hours, 0, -1)) {
        val features = ArrayBuffer[Double]()
        val labels = ArrayBuffer[Double]()
        // get current minute and recent two minutes
        for (index <- 0 to 2) {
          //当前时刻过去的时间那一分钟
          val tempOne = nowtimelong - 60 * 1000 * (i - index)
          val d = new Date(tempOne)
          val tempMinute = minuteFormat.format(d) //HHmm - 11:59
          //下一分钟
          val tempNext = tempOne - 60 * 1000 * (-1)
          val dNext = new Date(tempNext)
          val tempMinuteNext = minuteFormat.format(dNext) //HHmm

          for ((k, v) <- relations) {
            val map = v //map -- k:HHmm    v:Speed_count
            if (index == 2 && k == camera_id) {
              if (map.containsKey(tempMinuteNext)) {
                val info = map.get(tempMinuteNext).split("_")
                val avgSpeed = info(0).toFloat / info(1).toFloat
                labels += avgSpeed
              }
            }
            if (map.containsKey(tempMinute)) {
              val info = map.get(tempMinute).split("_")
              val avgSpeed = info(0).toFloat / info(1).toFloat
              features += avgSpeed
            } else {
              features += -1.0
            }
          }
        }

        if (labels.toArray.length == 1) {
          //array.head 返回数组第一个元素
          val label = (labels.toArray).head
          val record = LabeledPoint(if ((label.toInt / 10) < 10) (label.toInt / 10) else 10.0, Vectors.dense(features.toArray))
          //dataSet += record
          dataSet.appended(record)
        }
      }

      val data: RDD[LabeledPoint] = sc.parallelize(dataSet)

      // Split data into training (80%) and test (20%).
      //将data这个RDD随机分成 8:2两个RDD
      val splits = data.randomSplit(Array(0.8, 0.2))
      //构建训练集
      val training = splits(0)
      /**
       * 测试集的重要性：
       * 测试模型的准确度，防止模型出现过拟合的问题
       */
      val test = splits(1)

      if (!data.isEmpty()) {
        // 训练逻辑回归模型
        val model = new LogisticRegressionWithLBFGS()
          .setNumClasses(11)
          .setIntercept(true)
          .run(training)
        // 测试集测试模型
        val predictionAndLabels: RDD[(Double, Double)] = test.map { case LabeledPoint(label, features) =>
          val prediction = model.predict(features)
          (prediction, label)
        }

        predictionAndLabels.foreach(x => println("预测类别：" + x._1 + ",真实类别：" + x._2))

        // Get evaluation metrics. 得到评价指标
        val metrics: MulticlassMetrics = new MulticlassMetrics(predictionAndLabels)
        val precision = metrics.accuracy // 准确率
        println("Precision = " + precision)

        if (precision > 0.8) {
          val path = "hdfs://mynode1:8020/model/model_" + camera_id + "_" + nowtimelong
          model.save(sc, path)
          println("saved model to " + path)
          jedis.hset("model", camera_id, path)
        }
      }
    })
    RedisClient.pool.returnResource(jedis)
  }
}
