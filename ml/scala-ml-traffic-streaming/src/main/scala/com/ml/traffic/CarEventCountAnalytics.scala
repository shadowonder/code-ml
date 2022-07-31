package com.ml.traffic

import java.text.SimpleDateFormat
import java.util.Calendar

import net.sf.json.JSONObject
import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.SparkConf
import org.apache.spark.streaming._
import org.apache.spark.streaming.dstream.{DStream, InputDStream}
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010.{CanCommitOffsets, HasOffsetRanges, KafkaUtils, OffsetRange}

/**
 * 将每个卡扣的总速度_车辆数  存入redis中
 * 【yyyyMMdd_Monitor_id,HHmm,SpeedTotal_CarCount】
 */
object CarEventCountAnalytics {

  def main(args: Array[String]): Unit = {
    // Create a StreamingContext with the given master URL
    val conf = new SparkConf().setAppName("CarEventCountAnalytics")
    conf.set("spark.streaming.kafka.consumer.cache.enabled", "false")
    conf.setMaster("local[*]")
    val ssc = new StreamingContext(conf, Seconds(5))
    //    ssc.sparkContext.setCheckpointDir("xxx")
    // Kafka configurations
    val topics = Set("car_events")
    val brokers = "mynode1:9092,mynode2:9092,mynode3:9092"

    val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> brokers,
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> "group0903", //
      "auto.offset.reset" -> "earliest",
      "enable.auto.commit" -> (false: java.lang.Boolean) //默认是true
    )

    val dbIndex = 1

    // Create a direct stream
    val kafkaStream: InputDStream[ConsumerRecord[String, String]] = KafkaUtils.createDirectStream[String, String](
      ssc,
      PreferConsistent,
      Subscribe[String, String](topics, kafkaParams)
    )

    val events: DStream[JSONObject] = kafkaStream.map(line => {
      //JSONObject.fromObject 将string 转换成jsonObject
      val data: JSONObject = JSONObject.fromObject(line.value())
      println(data)
      data
    })

    /**
     * 输入(车辆速度,1), 输出(车辆速度的和,车辆总数)
     * carSpeed  K:monitor_id
     * V:(speedCount,carCount)
     */
    val carSpeed: DStream[(String, (Int, Int))] =
      events.map(jb => (jb.getString("camera_id"), jb.getInt("speed")))
        .mapValues((speed: Int) => (speed, 1))
        .reduceByKeyAndWindow((a: Tuple2[Int, Int], b: Tuple2[Int, Int]) => {
          (a._1 + b._1, a._2 + b._2)
        }, Seconds(60), Seconds(60))
    //              .reduceByKeyAndWindow((a:Tuple2[Int,Int], b:Tuple2[Int,Int]) => {(a._1 + b._1, a._2 + b._2)},(a:Tuple2[Int,Int], b:Tuple2[Int,Int]) => {(a._1 - b._1, a._2 - b._2)},Seconds(20),Seconds(10))

    carSpeed.foreachRDD(rdd => {
      rdd.foreachPartition(partitionOfRecords => {
        val jedis = RedisClient.pool.getResource
        partitionOfRecords.foreach(pair => {
          val camera_id = pair._1
          val speedTotal = pair._2._1
          val carCount = pair._2._2
          val now = Calendar.getInstance().getTime()
          // create the date/time formatters
          val dayFormat = new SimpleDateFormat("yyyyMMdd")
          val minuteFormat = new SimpleDateFormat("HHmm")
          val day = dayFormat.format(now) //20200903
          val time = minuteFormat.format(now) //1553
          if (carCount != 0 && speedTotal != 0) {
            jedis.select(dbIndex)
            //20200630_310999006001   -- k,v  -- (1130,200_5)
            jedis.hset(day + "_" + camera_id, time, speedTotal + "_" + carCount)
          }
        })
        RedisClient.pool.returnResource(jedis)
      })
    })

    /**
     * 异步更新offset
     */
    kafkaStream.foreachRDD { rdd =>
      val offsetRanges: Array[OffsetRange] = rdd.asInstanceOf[HasOffsetRanges].offsetRanges
      // some time later, after outputs have completed
      kafkaStream.asInstanceOf[CanCommitOffsets].commitAsync(offsetRanges)
    }
    ssc.start()
    ssc.awaitTermination()
  }
}