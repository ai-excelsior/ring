from typing import Any, Dict, List
import orjson
import pickle
import base64


def dumps(data, indent=True) -> bytes:
    option = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS

    if indent:
        option = option | orjson.OPT_INDENT_2

    return orjson.dumps(data, option=option)


def loads(s):
    return orjson.loads(s)


def pickle_dumps(obj) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def pick_loads(s: str) -> Any:
    return pickle.loads(base64.b64decode(s))


def dict_to_list(d: Dict) -> List:
    return [{"k": k, "v": v} for k, v in d.items()]


def kv_list_to_dict(l: List) -> Dict:
    return {item["k"]: item["v"] for item in l}
