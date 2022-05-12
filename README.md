# ring
Instructions are based on the fact that all code-relate files, including dockerfiles are correctly completed。
In other words, the only things left ARE to 
    a. build a image to run the model in docker, remotely
    b. apply k8s to do the conduction

## Prerequisite
- First login to our docker warehouse to obtain basic images 
- Secondly, Use `Makefile` to build and push specific images, for example `make build-docker-seq2seq-gpu`
- Then, put following enviroment variables in files, for example `.env_example` 

## Environment Variable
阿里 OSS 相关，用于保存训练结果:
- OSS_ACCESS_KEY_ID              # ID
- OSS_ACCESS_KEY_SECRET          # passwd
- OSS_ENDPOINT                   # address
InfluxDB 相关，用于保存预测结果：
- INFLUX_ENDPOINT                # address
- INFLUX_TOKEN                   # token
- INFLUX_ORG                     # chosen orgnization
- INFLUX_PREDICTION_BUCKET_NAME  # bucket to put prediction results
- INFLUX_LOG_BUCKET_NAME         # bucket to put training and validation logs

## Run the specific docker-image directly
E.g.: Using docker image to train on air-passengers dataset:

```bash
docker run -it --rm --env-file=.env_example \                                              # declare env file
    code.unianalysis.com:5050/unianalysis/ring/seq2seq-gpu:0.0.1  \                        # declare images
    train \                                                                                # args declare in `ENTRYPOINT` in dockerfile
    --data_train=oss://aiexcelsior-shanghai-test/liyu_test_data/air_passengers_train.csv \
    --data_val=oss://aiexcelsior-shanghai-test/liyu_test_data/air_passengers_val.csv \
    --data_cfg=oss://aiexcelsior-shanghai-test/liyu_test_data/air_passengers-config.json \
    --early_stopping_patience=1
    --max_epochs=2
    --lr=0.01
```

## Conduct jobs using k8s
- Follow instructions in `如何基于K8S和KubeFlow使用服务器GPU资源`