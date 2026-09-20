import os
import csv
from pathlib import Path

os.environ["SPARK_LOCAL_DIRS"] = r"C:\SparkTemp"
os.makedirs(r"C:\SparkTemp", exist_ok=True)

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_DIR / "results" / "cleaned_traffic_data.csv")
OUTPUT_PATH = PROJECT_DIR / "results" / "spark_hourly_analysis.csv"


def create_spark_session():
    return (
        SparkSession.builder
        .appName("SmartTrafficManagement")
        .master("local[*]")
        .getOrCreate()
    )


def main():

    spark = create_spark_session()

    print("===== PYSPARK TRAFFIC PROCESSING =====")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(DATA_PATH)
    )

    print("\nSchema:")
    df.printSchema()

    total_records = df.count()

    print("\nTotal records:")
    print(total_records)

    print("\nPerforming Spark aggregation...")

    hourly_analysis = (
        df.groupBy("Hour Of Day")
        .agg(
            avg("Traffic Density").alias("Average Traffic Density"),
            avg("Speed").alias("Average Speed"),
            count("*").alias("Record Count")
        )
        .orderBy("Hour Of Day")
    )

    print("\n===== HOURLY TRAFFIC ANALYSIS =====")
    hourly_analysis.show(24, truncate=False)

    print("\nExporting 24-row Spark result...")

    rows = hourly_analysis.collect()

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(hourly_analysis.columns)

        for row in rows:
            writer.writerow(row)

    print("\nSpark aggregation result saved to:")
    print(OUTPUT_PATH)

    spark.stop()

    print("\nPySpark processing completed successfully.")


if __name__ == "__main__":
    main()