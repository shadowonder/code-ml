package com.ml.traffic

import redis.clients.jedis.JedisPool

object RedisClient extends Serializable {
  val redisHost = "192.168.10.20"
  val redisPort = 6379
  val redisTimeout = 30000
  /**
   * JedisPool是一个连接池，既可以保证线程安全，又可以保证了较高的效率。 
   */
  lazy val pool = new JedisPool(redisHost, redisPort)

  //  lazy val hook = new Thread {
  //    override def run = {
  //      println("Execute hook thread: " + this)
  //      pool.destroy()
  //    }
  //  }
  //  sys.addShutdownHook(hook.run)
}