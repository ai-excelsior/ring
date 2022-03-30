import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd

# initialize influxdb client
bucket = "sample_test"
url = "http://192.168.1.58:8086/"
token = "J49Ebh3ImtDuiC4dPm44N1qEGfo3-zRJk2kkBVnuiJCNNoMYdIl2oQ8f55IOLgDwFIRZaEpDKmsKGL6cQM3KHw=="
org = "unianalysis"
client = influxdb_client.InfluxDBClient(
    url=url,
    token=token,
    org=org,
)
# # initialize write client
# write_api = client.write_api(write_options=SYNCHRONOUS)
# In InfluxDB, a point represents a single data record, similar to a row in a SQL database table.
# p = influxdb_client.Point("my_measurement").tag("location", "Prague").field("temperature", 25.3)
# write_api.write(bucket=bucket, org=org, record=p)

# initialize query client
query_api = client.query_api()
query = 'from(bucket:"sample_test")\
|> range(start:2016-05-22T23:30:00Z,stop: now())\
|> filter(fn:(r) => r._measurement == "djia_stocks" and r._field=="Close")' \
result = query_api.query_data_frame(query=query)  # _time, _value, `tag`, _field
# result = [result] if not isinstance(result, list) else result  # each `tag` each pd
# dff = pd.DataFrame()
# for i in result:
#     df = i[["_time", "_value", "_field", "Name"]]
#     two_level_index_series = df.set_index(["_time", "_field"])["_value"]
#     new_df = two_level_index_series.unstack()
#     new_df = new_df.rename_axis(columns=None)
#     new_df = new_df.reset_index()
#     new_df["Name"] = df["Name"][0]
#     if len(dff) == 0:
#         dff = new_df
#     elif i["Name"][0] == dff["Name"][0]:
#         dff = pd.merge(left=dff, right=new_df, on=["_time", "Name"], how="right")
#     else:
#         dff.to_csv("~/Desktop/m4-dataset/test_" + dff["Name"][0] + ".csv", index=None)
#         dff = new_df

# dff.to_csv("~/Desktop/m4-dataset/test_" + dff["Name"][0] + ".csv", index=None)
