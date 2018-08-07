## prerequisite
- java 8
- maven

## prepare
```bash
mkdir -p data/train
mkdir -p data/test
cp data/data.csv data/test/data.csv
# more data for train
for i in {1..100};do cat data/data.csv >> data/train/data_dup.csv; done
```

## package
```bash
mvn clean package -Djavacpp.platform=linux-x86_64
```

## launch
```bash
java -Dorg.bytedeco.javacpp.maxbytes=60G -Dorg.bytedeco.javacpp.maxphysicalbytes=60G -XX:+UseG1GC \
   -jar target/tuning-1.0.0-shaded.jar \
   --input-path data  \
   --output-dir-path tuning_result \
   --number-of-epochs 10 &> tuning.log &
 
```
