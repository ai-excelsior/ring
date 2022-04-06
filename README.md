# ring

Using docker image to train on air-passengers dataset:

```bash
docker run -it --rm --env-file=env code.unianalysis.com:5050/unianalysis/ring/seq2seq-gpu:0.0.1 train \
    --data_train=oss://aiexcelsior-shanghai-test/liyu_test_data/air_passengers_train.csv \
    --data_val=oss://aiexcelsior-shanghai-test/liyu_test_data/air_passengers_val.csv \
    --data_cfg=oss://aiexcelsior-shanghai-test/liyu_test_data/air_passengers-config.json \
    --early_stopping_patience=1
```
