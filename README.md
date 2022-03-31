# ring

Using docker image to train on air-passengers dataset:

```bash
docker run -it --rm --env-file=.env seq2seq train --data_train=oss://aiexcelsior-shanghai-test/liyu_test_data/air_passengers_train.csv \
    --data_val=oss://aiexcelsior-shanghai-test/liyu_test_data/air_passengers_val.csv \
    --data_cfg=oss://aiexcelsior-shanghai-test/liyu_test_data/air_passengers-config.json \
    --model_state=air_passengers.zip \
    --early_stopping_patience=1
```
