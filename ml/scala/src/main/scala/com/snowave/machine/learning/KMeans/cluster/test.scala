package com.snowave.machine.learning.KMeans.cluster

import org.apache.lucene.analysis.TokenStream
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.feature.{HashingTF, IDF, IDFModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.wltea.analyzer.lucene.IKAnalyzer

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object test {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("KMeans1").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("Error")
    // 读取test文件
    val rdd = sc.textFile("./sample-data/data/kmean-clusters/testdata.txt", 8)
    var wordRDD = rdd.mapPartitions(iterator => {
      val list = new ListBuffer[(String, ArrayBuffer[String])]
      while (iterator.hasNext) {
        val analyzer = new IKAnalyzer(false)
        val line = iterator.next()
        val textArr = line.split("\t")
        val id = textArr(0)
        val text = textArr(1)
        val ts: TokenStream = analyzer.tokenStream("", text)
        val term: CharTermAttribute = ts.getAttribute(classOf[CharTermAttribute])
        ts.reset()
        val arr = new ArrayBuffer[String]
        while (ts.incrementToken()) {
          arr.+=(term.toString())
        }

        list.append((id, arr))
        analyzer.close()
      }
      list.iterator
    })
    wordRDD = wordRDD.cache()
    println("********* wordRDD ***********************")
    wordRDD.foreach(println)
    // 输出:
    // ********* wordRDD ***********************
    // 如果需要使用特殊词典, 我们需要关闭智能分词
    // Load extended dictionary:ext.dic  // 这里使用了ext.dic, 可以配置这个文件用来保护我们需要的词不被切割
    // Load stopwords dictionary:stopword.dic  // stopwords.dic 用来存放我们不想要的词, 这些词就会被切分
    // (3794020835114250,ArrayBuffer(我要, 天, 天和, 当家, 看, 日出))
    // (3794020835114251,ArrayBuffer(我要, 天, 天和, 当家, 看, 日出))
    // (3794020835114249,ArrayBuffer(九阳, 必须, 是, 其中之一, 的, 其中之一))


    val hashingTF: HashingTF = new HashingTF(1000)
    val tfRDD = wordRDD.map(x => {
      (x._1, hashingTF.transform(x._2))
    })
    println("*********tfRDD***********************")
    tfRDD.foreach(println)

    val idf: IDFModel = new IDF().fit(tfRDD.map(_._2))

    val tfIdfs: RDD[(String, Vector)] = tfRDD.mapValues(idf.transform)
    println("===========tfIdfs=================")
    tfIdfs.foreach(println)

    wordRDD = wordRDD.mapValues(buffer => {
      buffer.distinct.sortBy(item => {
        hashingTF.indexOf(item)
      })
    })
    println("===========wordRDD=================")
    wordRDD.foreach(println)

    //设置聚类个数
    val kcluster = 2
    val kmeans = new KMeans()
    kmeans.setK(kcluster)
    //使用的是kemans++算法来训练模型  "random"|"k-means||"
    kmeans.setInitializationMode("k-means||")
    //设置最大迭代次数
    kmeans.setMaxIterations(100)
    //训练模型
    val kmeansModel: KMeansModel = kmeans.run(tfIdfs.map(_._2))
    //    kmeansModel.save(sc, "d:/model001")
    //打印模型的20个中心点
    val centers = kmeansModel.clusterCenters
    println("中心点：")
    centers.foreach(println)
    //    println(kmeansModel.clusterCenters)

    /**
     * 模型预测
     */
    val modelBroadcast = sc.broadcast(kmeansModel)
    /**
     * predicetionRDD KV格式的RDD
     * K：微博ID
     * V：分类号
     */
    val predicetionRDD = tfIdfs.mapValues(sample => {
      val model = modelBroadcast.value
      model.predict(sample)
    })
    //    predicetionRDD.saveAsTextFile("d:/resultttt")

    /**
     * 总结预测结果
     * tfIdfs2wordsRDD:kv格式的RDD
     * K：微博ID
     * V：二元组(Vector(tfidf1,tfidf2....),ArrayBuffer(word,word,word....))
     */
    val tfIdfs2wordsRDD = tfIdfs.join(wordRDD)
    /**
     * result:KV
     * K：微博ID
     * V:(类别号，(Vector(tfidf1,tfidf2....),ArrayBuffer(word,word,word....)))
     */
    val result = predicetionRDD.join(tfIdfs2wordsRDD)

    /**
     * 查看0号类别中tf-idf比较高的单词，能代表这类的主题
     */
    result
      .filter(x => x._2._1 == 1)
      .flatMap(line => {

        val tfIdfV: Vector = line._2._2._1
        val words: ArrayBuffer[String] = line._2._2._2
        val list = new ListBuffer[(Double, String)]

        for (i <- 0 until words.length) {
          //hashingTF.indexOf(words(i)) 当前单词在1000个向量中的位置
          list.append((tfIdfV(hashingTF.indexOf(words(i))), words(i)))
        }
        list
      })
      .sortBy(x => x._1, false)
      .map(_._2)
      .filter(_.length() > 1).distinct()
      .take(30).foreach(println)
    sc.stop()
  }

}