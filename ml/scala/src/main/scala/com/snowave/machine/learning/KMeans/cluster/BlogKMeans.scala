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

/**
 * <!-- 项目工具 -->
 * <dependency>
 * <groupId>com.github.magese</groupId>
 * <artifactId>ik-analyzer</artifactId>
 * <version>8.5.0</version>
 * </dependency>
 *
 * 原始的数据:
 *
 * 3793992720744105	#九阳有礼 无需多滤#陷入被窝温柔乡，起床靠毅力？九阳免滤豆浆机C668SG耀世首发！智能预约免过滤，贴心配置强到飞起，让你再续温柔一小时！真的很需要这款九阳豆浆机，这样就可以和小宝贝多待会！@高海澄知 @历衔枫 @郭河九元
 * 3793993084926422	#谢谢你陪我走过2014#好吧，这一年马上就要过去了，同样这一年有欢笑，有泪水，但更多的还是幸福。虽然我知道我很任性[纠结]，但宝宝姐姐老婆还是对我超级好好好好好好[群体围观]，希望我明年能乖点，听点话 @九阳 @瑷o詠a国际范
 * 3793993291060111	跨年啦。小伙伴们，新年快乐～[笑哈哈][笑哈哈][笑哈哈]@美的电饭煲官方微博 @美的生活电器 @九阳 @SKG互联网家电 @中国电信湖北客服
 * 3793993588106975	我的胆有0.9斤，我想要3.1斤重的铁釜，有份量才够胆量！九阳Alva0716
 * 3793995102741635	《太上青玄慈悲太乙救苦天尊寶懺》 - 起讚   元始運元  神運元神 &#57359;化太一尊    九陽天上布恩綸 手內楊枝 遍灑甘露春   大眾悉朝真 群荷深仁 朵朵擁祥雲 大...  (来自 @头条博客) -  顶礼太上青玄慈悲太乙救苦天尊 http://t.cn/zYwwlSY
 * 3793995370610238	#九阳有礼 无需多滤#新年交好运！有了九阳，让生活免滤无忧！@誰能許诺給我一世柔情 @索心进 @错爱990
 * 3793995484592300	#谢谢你陪我走过2014#2014年将至，希望能中一个好东西来送给我的家人。@九阳 @枫叶红了112
 * 37939954845923011	#谢谢你陪我走过2014#2014年将至，希望能中一个好东西来送给我的家人。@九阳 @枫叶红了112
 * 3793995781905340	免过滤，更顺滑，#九阳有礼 无需多滤# 更多营养更安心！@princess佳妮昂 @木凝眉 @单纯会让人受伤航
 * 3793996277455995	#谢谢你陪我走过2014#2014年将至，希望能中一个好东西来送给我的家人。@九阳 @枫叶红了112
 * 3793996323668014	#谢谢你陪我走过2014#2014年将至，希望能中一个好东西来送给我的家人。@九阳 @枫叶红了112
 * 3793996390629648	#谢谢你陪我走过2014#2014年将至，希望能中一个好东西来送给我的家人。@九阳 @枫叶红了112
 */
object BlogKMeans {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("BlogKMeans").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val rdd = sc.textFile("./sample-data/data/kmean-clusters/original.txt", 8)

    // mapPartitions 会输入一个iterator 然后返回一个iterator
    var wordRDD: RDD[(String, ArrayBuffer[String])] = rdd.mapPartitions(iterator => {
      // 返回值 list.iterator()
      // 这里的key = 微博的id
      // value是分词器切分的词
      val list = new ListBuffer[(String, ArrayBuffer[String])]
      while (iterator.hasNext) { // 获取一行
        //创建分词对象   IKAnalyzer支持两种分词模式：最细粒度和智能分词模式，如果构造函数参数为false，那么使用最细粒度分词。
        val analyzer = new IKAnalyzer(true) // 智能分词为true, 一般为true, 更精简
        val line = iterator.next()
        val textArr = line.split("\t")
        val id = textArr(0)
        val text = textArr(1)
        //分词     第一个参数只是标识性，没有实际作用，第二个读取的数据
        val ts: TokenStream = analyzer.tokenStream("", text)
        //得到相应词汇的内容
        val term: CharTermAttribute = ts.getAttribute(classOf[CharTermAttribute])
        //重置分词器，使得tokenstream可以重新返回各个分词. 如果不reset就获取不了
        ts.reset()
        val arr = new ArrayBuffer[String] // 输出的数据
        //遍历分词数据
        while (ts.incrementToken()) {
          arr.+=(term.toString())
        }

        list.append((id, arr))
        analyzer.close()
      }
      list.iterator
    })

    /**
     * wordRDD 是一个KV格式的RDD
     * K:微博ID
     * V:微博内容分词后的结果 ArrayBuffer
     */
    // (3794020835114250,ArrayBuffer(我要, 天, 天和, 当家, 看, 日出))
    // (3794020835114251,ArrayBuffer(我要, 天, 天和, 当家, 看, 日出))
    // (3794020835114249,ArrayBuffer(九阳, 必须, 是, 其中之一, 的, 其中之一))
    wordRDD = wordRDD.cache()
    //    val allDicCount = wordRDD.flatMap(one=>{one._2}).distinct().count()
    /**
     * HashingTF 使用hash表来存储分词
     * HashingTF 是一个Transformer 转换器，在文本处理中，接收词条的集合然后把这些集合转化成固定长度的特征向量,这个算法在哈希的同时会统计各个词条的词频
     * 提高hash表的桶数，默认特征维度是 2的20次方 = 1,048,576
     * 1000:只是计算每篇微博中1000个单词的词频
     *
     * 使用hashTF的目的是方便定义向量, 如果使用原始词频那么就会有100万个features
     */
    val hashingTF: HashingTF = new HashingTF(1000)

    /**
     * 对每一个词进行hash, 然后放入hash桶内, 获得了每一个次的向量也就是TF
     * *********tfRDD*******
     * (3794020835114250,(1000,[14,22,281,509,564,947,950,957],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]))
     * (3794020835114249,(1000,[132,215,367,369,633,660,738,750,808,996],[1.0,1.0,1.0,2.0,1.0,2.0,1.0,1.0,2.0,2.0]))
     * (3794020835114251,(1000,[14,22,281,509,564,947,950,957],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]))
     */
    val tfRDD: RDD[(String, Vector)] = wordRDD.map(x => {
      (x._1, hashingTF.transform(x._2))
    })

    /**
     * tfRDD
     * K:微博ID
     * V:Vector（tf，tf，tf.....）
     *
     * hashingTF.transform(x._2)
     * 按照hashingTF规则 计算分词频数（TF）
     */
    //    tfRDD.foreach(println)

    /**
     * 得到IDFModel，要计算每个单词在整个语料库中的IDF
     * IDF是一个 Estimator 评价器，在一个数据集上应用它的fit（）方法，产生一个IDFModel。 该IDFModel 接收特征向量（由HashingTF产生）
     * new IDF().fit(tfRDD.map(_._2)) 就是在组织训练这个评价器，让评价器知道语料库中有那些个词块，方便计算IDF
     *
     * fit(tfRdd.map(_._2)) 就是将tf中的结果放入IDF中训练. 得出IDF模型
     */
    val idf: IDFModel = new IDF().fit(tfRDD.map(_._2))

    /**
     * K:微博 ID
     * V:每一个单词的TF-IDF值
     * tfIdfs这个RDD中的Vector就是训练模型的训练集
     * 计算TFIDF值
     *
     * 使用IDF模型将TF的value进行转换. 得到一个kv的rdd, key就是文本ID, value就是tfidf值
     * ===========tfIdfs===========
     * 3794020835114251,(1000,[14,22,281,509,564,947,950,957],[0.28768207245178085,0.28768207245178085,0.28768207245178085,0.28768207245178085,0.28768207245178085,0.28768207245178085,0.28768207245178085,0.28768207245178085]))
     * (3794020835114249,(1000,[132,215,367,369,633,660,738,750,808,996],[0.6931471805599453,0.6931471805599453,0.6931471805599453,1.3862943611198906,0.6931471805599453,1.3862943611198906,0.6931471805599453,0.6931471805599453,1.3862943611198906,1.3862943611198906]))
     * (3794020835114250,(1000,[14,22,281,509,564,947,950,957],[0.28768207245178085,0.28768207245178085,0.28768207245178085,0.28768207245178085,0.28768207245178085,0.28768207245178085,0.28768207245178085,0.28768207245178085]))
     */
    val tfIdfs: RDD[(String, Vector)] = tfRDD.mapValues(idf.transform(_))

    /**
     * 如何知道tfIdfs 中Vector中的每个词对应的分词？？？
     *
     * wordRDD [微博ID,切出的来的单词数组]
     *
     * hashingTF.indexOf(item)  方法传入一个单词item，返回当前词组item 在hashingTF转换器写对应的分区号
     * 以下的做法就是按照每个词由hashingTF 映射的分区号由小到大排序，得到的每个词组对应以上得到的tfIdfs 值的顺序
     *
     * 也就是 将单词按照tfidf进行排序. 因此每一个单词的index和上面tfidf的顺序是相同的
     *
     * (3794020835114249,ArrayBuffer(阳, 九阳, 九, 之一, 必须, 一, 是, 的, 其中, 其中之一))
     * (3794020835114250,ArrayBuffer(天和, 要, 日出, 当家, 看, 天天, 我要, 我))
     * (3794020835114251,ArrayBuffer(天和, 要, 日出, 当家, 看, 天天, 我要, 我))
     */
    wordRDD = wordRDD.mapValues(buffer => {
      buffer.distinct.sortBy(item => {
        hashingTF.indexOf(item)
      })
    })


    /**
     * 执行 Kmean
     */
    //设置聚类个数
    val kcluster = 20
    val kmeans = new KMeans()
    kmeans.setK(kcluster)
    //使用的是kemans++算法来训练模型  "random"|"k-means||"
    kmeans.setInitializationMode("k-means||")
    //设置最大迭代次数
    kmeans.setMaxIterations(1000)
    //训练模型
    val kmeansModel: KMeansModel = kmeans.run(tfIdfs.map(_._2)) // 将value传入进来, 也就是所有的features
    //    kmeansModel.save(sc, "d:/model001")
    //打印模型的20个中心点
    println(kmeansModel.clusterCenters)

    /**
     * 模型预测
     */
    val modelBroadcast = sc.broadcast(kmeansModel)
    /**
     * predicetionRDD KV格式的RDD
     * K：微博ID
     * V：分类号
     */
    val predicetionRDD: RDD[(String, Int)] = tfIdfs.mapValues(vetor => {
      val model = modelBroadcast.value
      model.predict(vetor) // 得到分类的类别编号
    })
    /**
     * 总结预测结果
     * tfIdfs2wordsRDD:kv格式的RDD
     * K：微博ID
     * V：二元组(Vector(tfidf1,tfidf2....),ArrayBuffer(word,word,word....))
     */
    val tfIdfs2wordsRDD: RDD[(String, (Vector, ArrayBuffer[String]))] = tfIdfs.join(wordRDD)
    /**
     * 将全部的预测结果和类别编号join一下
     * result:KV
     * K：微博ID
     * V:(类别号，而为元祖(key: tfidf, value: 每一个单词))
     */
    val result: RDD[(String, (Int, (Vector, ArrayBuffer[String])))] = predicetionRDD.join(tfIdfs2wordsRDD)

    /**
     * 查看1号类别中tf-idf比较高的单词，能代表这类的主题
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
        list // 得出kv格式的list, k就是tfidf value就是词
      })
      .sortBy(x => x._1, false) // 对tfidf进行排序
      .map(_._2).distinct() // 对词进行去重
      .filter(one => {
        one.length > 1 // 一些单个的词删除掉
      })
      .take(30).foreach(println)
    sc.stop()
  }
}
