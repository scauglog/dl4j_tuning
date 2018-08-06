## more data
```bash
for i in {1..100};do cat data.csv >> data/train/data_dup.csv; done
for i in {1..10};do cp data/train/data_dup.csv data/train/data_dup_$i.csv; done

```


## launch
```bash
export JAVA_HOME=/opt/java/current
export SPARK_HOME=/opt/spark/current
SPARK_SUBMIT=$SPARK_HOME/bin/spark-submit
SPARK_OPTIONS="--master local --deploy-mode client --driver-memory 20G --num-executors 3 --executor-memory 10g --executor-cores 1"
$SPARK_SUBMIT $SPARK_OPTIONS --conf "spark.driver.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=60G -Dorg.bytedeco.javacpp.maxphysicalbytes=60G -XX:+UseG1GC" --name "ClicksAndCostPredictions" \
   --class aperture.tuning.predictions.tuning.TuningMain \
   --conf "spark.sql.shuffle.partitions=24" \
   tuning-1.0.0-shaded.jar \
   --input-path data  \
   --output-dir-path tuning_result \
   --number-of-epochs 10 &> tuning.log &

```