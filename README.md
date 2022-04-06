# ring

## Environment Variable

阿里 OSS 相关，用于保存训练结果:

- OSS_ACCESS_KEY_ID
- OSS_ACCESS_KEY_SECRET
- OSS_ENDPOINT

InfluxDB 相关，用于保存预测结果：

- INFLUX_ENDPOINT
- INFLUX_TOKEN
- INFLUX_ORG
- INFLUX_PREDICTION_BUCKET_NAME

Using docker image to train on air-passengers dataset:

```bash
docker run -it --rm --env-file=env code.unianalysis.com:5050/unianalysis/ring/seq2seq-gpu:0.0.1 train \
    --data_train=oss://aiexcelsior-shanghai-test/liyu_test_data/air_passengers_train.csv \
    --data_val=oss://aiexcelsior-shanghai-test/liyu_test_data/air_passengers_val.csv \
    --data_cfg=oss://aiexcelsior-shanghai-test/liyu_test_data/air_passengers-config.json \
    --early_stopping_patience=1
```
