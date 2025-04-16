from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from pyspark import SparkConf

schema = StructType([
    StructField("location", StringType()),
    StructField("date", StringType()),
    StructField("total_cases", DoubleType()),
    StructField("new_cases", DoubleType()),
    StructField("total_deaths", DoubleType()),
    StructField("new_deaths", DoubleType()),
    StructField("total_vaccinations", DoubleType()),
    StructField("people_vaccinated", DoubleType()),
    StructField("people_fully_vaccinated", DoubleType()),
    StructField("new_vaccinations", DoubleType()),
    StructField("life_expectancy", DoubleType()),
    StructField("excess_mortality", DoubleType())
])

def create_spark_session():
    """
    Crée et retourne une session Spark.
    """
    conf = SparkConf()

    conf.set("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,mysql:mysql-connector-java:8.0.23")


    conf.set("spark.sql.streaming.checkpointLocation", "hdfs://localhost:8020/checkpoints/mobility")
    conf.set("spark.hadoop.fs.defaultFS", "hdfs://localhost:8020")
    conf.set("spark.driver.host", "localhost")
    conf.set("spark.driver.bindAddress", "0.0.0.0")
    conf.set("spark.ui.port", "4041")
    conf.set("spark.kafka.consumer.request.timeout.ms", "60000")
    conf.set("spark.kafka.consumer.session.timeout.ms", "30000")
    conf.set("spark.sql.streaming.kafka.consumer.poll.ms", "60000")
    return SparkSession.builder \
        .appName("Covid19Data") \
        .config(conf=conf) \
        .getOrCreate()

def process_kafka_data(spark):
    """
    Lit les données COVID-19 depuis Kafka, les traite et les écrit dans HDFS.
    """
    df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092")  \
        .option("subscribe", "pandemic_covid") \
        .option("startingOffsets", "earliest") \
        .option("maxOffsetsPerTrigger", 1000000000) \
        .load()

    print("Read stream")

    df = df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    df = df.withColumn("date", to_timestamp(col("date"), "yyyy-MM-dd"))

    query = df.writeStream \
        .partitionBy("date") \
        .format("parquet") \
        .option("path", "hdfs://localhost:8020/data/covid19") \
        .outputMode("append") \
        .trigger(once=True) \
        .start()
    query.awaitTermination()
    return query



if __name__ == "__main__":
    print("Creating Session...")
    spark = create_spark_session()
    print("Session created. Pulling data...")
    query = process_kafka_data(spark)
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("Arrêt du traitement Spark Streaming...")
        query.stop()
        spark.stop()
    print("Fin du programme.")