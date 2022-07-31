package com.ml.traffic

import java.util.Properties

import net.sf.json.JSONObject
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import org.apache.spark.{SparkConf, SparkContext}

/**
 * 生产者
 * 向kafka car_events中生产数据
 * 案例数据:
 * '310999003001', '3109990030010220140820141230292','00000000','','2014-08-20 14:09:35','0',255,'SN',  0.00,'4','','310999','310999003001','02','','','2','','','2014-08-20 14:12:30','2014-08-20 14:16:13',0,0,'2014-08-21 18:50:05','','',' '
 * '310999003102', '3109990031020220140820141230266','粤BT96V3','','2014-08-20 14:09:35','0',21,'NS',  0.00,'2','','310999','310999003102','02','','','2','','','2014-08-20 14:12:30','2014-08-20 14:16:13',0,0,'2014-08-21 18:50:05','','',' '
 * '310999000106', '3109990001060120140820141230316','沪F35253','','2014-08-20 14:09:35','0',57,'OR',  0.00,'2','','310999','310999000106','01','','','2','','','2014-08-20 14:12:30','2014-08-20 14:16:13',0,0,'2014-08-21 18:50:05','','',' '
 *
 * 生成数据:
 * {"camera_id":"'310999017502'","car_id":"'沪D71019'","speed":"42"}
 * {"camera_id":"'310999017201'","car_id":"'沪FW2590'","speed":"33"}
 * {"camera_id":"'310999020506'","car_id":"'浙A7ZD32'","speed":"47"}
 * {"camera_id":"'310999021006'","car_id":"'皖AV799A'","speed":"35"}
 * {"camera_id":"'310999017903'","car_id":"'沪A1U268'","speed":"54"}
 * {"camera_id":"'310999001903'","car_id":"'苏A1TK62'","speed":"46"}
 * {"camera_id":"'310999003601'","car_id":"'沪K89757'","speed":"31"}
 * {"camera_id":"'310999003601'","car_id":"'沪G12557'","speed":"49"}
 * {"camera_id":"'310999004002'","car_id":"'沪B9F995'","speed":"53"}
 * {"camera_id":"'310999004002'","car_id":"'川ANK539'","speed":"45"}
 * {"camera_id":"'310999005803'","car_id":"'沪CCG926'","speed":"48"}
 * {"camera_id":"'310999008105'","car_id":"'浙KL5720'","speed":"50"}
 */
object KafkaEventProducer {
  def main(args: Array[String]): Unit = {
    val topic = "car_events" // kafka的topic

    val props = new Properties()
    props.put("bootstrap.servers", "192.168.10.20:9092")
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")

    // 创建producer对象
    val producer = new KafkaProducer[String, String](props)

    val sparkConf = new SparkConf().setAppName("traffic data").setMaster("local[4]")
    val sc = new SparkContext(sparkConf)

    // 读取文件
    val records: Array[Array[String]] = sc.textFile("./data/carFlow_all_column_test.txt")
      .filter(!_.startsWith(";")) // 不能分好开头
      .filter(one => {
        !"'00000000'".equals(one.split(",")(2)) // 过滤掉车牌号为0000000的
      })
      .filter(_.split(",")(6).toInt != 255) // 过滤掉车辆速度为255 和 0
      .filter(_.split(",")(6).toInt != 0)
      .map(_.split(",")).collect()

    // 发送1000次
    for (i <- 1 to 1000) {
      for (recordArr <- records) {
        // prepare event data
        val event = new JSONObject()
        event.put("camera_id", recordArr(0))
        event.put("car_id", recordArr(2))
        event.put("speed", recordArr(6))
        // produce event message
        producer.send(new ProducerRecord[String, String](topic, event.toString))
        Thread.sleep(200)
      }
    }
    sc.stop
  }
}