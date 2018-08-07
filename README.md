## prerequisite
- spark 2.1.1
- java 8
- maven

## prepare
```bash
mkdir -p data/train
mkdir -p data/test
cp data/data.csv data/test/data.csv
# more data for train
for i in {1..100};do cat data/data.csv >> data/train/data_dup.csv; done
for i in {1..10};do cp data/train/data_dup.csv data/train/data_dup_$i.csv; done
```

## export java and spark home
```bash
export JAVA_HOME={{PATH_TO_JAVA8_FOLDER}}
export SPARK_HOME={{PATH_TO_SPARK_FOLDER}}
```

## package
```bash
mvn clean package -Djavacpp.platform=linux-x86_64
```

## launch
check that `driver-memory` + (`num-executors` * `executor-memory`) is less than your total memory

```bash
SPARK_SUBMIT=$SPARK_HOME/bin/spark-submit
SPARK_OPTIONS="--master local --deploy-mode client --driver-memory 20G --num-executors 3 --executor-memory 10g --executor-cores 1"
$SPARK_SUBMIT $SPARK_OPTIONS --conf "spark.driver.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=60G -Dorg.bytedeco.javacpp.maxphysicalbytes=60G -XX:+UseG1GC" --name "ClicksAndCostPredictions" \
   --class aperture.tuning.predictions.tuning.TuningMain \
   --conf "spark.sql.shuffle.partitions=24" \
   target/tuning-1.0.0-shaded.jar \
   --input-path data  \
   --output-dir-path tuning_result \
   --number-of-epochs 10 &> tuning.log &

```
