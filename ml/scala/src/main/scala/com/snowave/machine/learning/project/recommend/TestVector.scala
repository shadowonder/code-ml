package com.snowave.machine.learning.project.recommend

import org.apache.spark.mllib.linalg.SparseVector

object TestVector {
  def main(args: Array[String]): Unit = {
    //    while(true){
    //
    //    	println(2*(2*random()).toInt)
    //    }
    //    println(2*(2*random()).toInt-1)


    //    val conf = new SparkConf().setMaster("local").setAppName("test")
    //    val sc = new SparkContext(conf)
    //    val rdd = sc.makeRDD(List("bjsxt","shsxt","gzsxt"))
    //    val result = rdd.zipWithIndex();
    //    result.foreach(println)
    //    val zipWithIndexRDD = rdd.zipWithIndex()
    //    zipWithIndexRDD.foreach(println)

    //    val array: Array[Int] = Array.fill(3)(100)
    //    array.foreach(println)

    /**
     * 稀疏向量, 将水平数据网格化
     * 比如数据为(a,1),(b,1),(e,1)...进入为
     * a  b  c  d  e  f  g  h  i  j
     * 1  1  0  0  1  0  0  0  0  0
     *
     * 第一个参数为size, 就是有多少个feature
     * 第二个参数为将要更新参数的index坐标, 此处更新的是1,3,5 也就是 b,d,f
     * 第三个参数表示的是对于第二个参数给的坐标, 赋予的值是多少, 比如b=100,d=200 等
     */
    val vector = new SparseVector(10, Array(1, 3, 5), Array(100, 200, 300))
    println(vector.toDense)

    Array.fill(3)("Hello") // 创建一个满是"hello"的array

    //zip  
    //    val rdd1 = sc.makeRDD(1 to 10)
    //    val rdd2 = sc.makeRDD(101 to 110)
    //    val rdd3 = rdd1.zip(rdd2)
    //    rdd3.foreach(println)

    //    sc.stop()
  }
}