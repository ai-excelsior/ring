from typing import Union


def get_train_parser(subparsers):
    parser = subparsers.add_parser("train", help="train a model")
    # parser = subparser
    parser.add_argument("--data_cfg", type=str, required=True, help="The data config, s3 address")
    parser.add_argument("--data_train", type=str, required=True, help="The train data source, s3 address")
    parser.add_argument("--data_val", type=str, required=True, help="The val data source, s3 address")
    parser.add_argument(
        "--load_state",
        type=str,
        default=None,
        help="Load a pre-training load_state to continue refining, s3 address",
    )
    parser.add_argument(
        "--save_state",
        type=str,
        default=None,
        help="Upload the training save_state for validation, prediction or later refining, oss address",
    )

    # common trainer config
    parser.add_argument("--metric", type=str, default="MSE")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--weight_decay", type=str, default=0)
    parser.add_argument("--max_clip_grad_norm", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--logger_mode", type=str, default="local")
    parser.add_argument("--task_id", type=str, default="task_default_none")
    return parser


def get_validate_parser(subparsers):
    parser = subparsers.add_parser("validate", help="validate a model, get validation metrics")
    parser.add_argument("--data_cfg", type=str, required=True, help="The data config, s3 address")
    parser.add_argument("--data_val", type=str, required=True, help="The val data source, s3 address")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--load_state",
        type=str,
        default=None,
        help="Load a pre-training model_state to do validation, s3 address",
    )
    parser.add_argument(
        "--begin_point",
        type=str,
        default=None,
        help="Identify the begin point of validation, both datetime or int",
    )
    return parser


def get_predict_parser(subparsers):
    parser = subparsers.add_parser("predict", help="predict the last part of the given data")
    parser.add_argument("--data_cfg", type=str, required=True, help="The data config, s3 address")
    parser.add_argument("--data", type=str, required=True, help="The data source, s3 address")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--load_state",
        type=str,
        default=None,
        help="Load a pre-training model_state to do prediction, s3 address",
    )

    parser.add_argument(
        "--measurement",
        type=str,
        required=False,
        help="The measurement name when saving predictions result to influxdb",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        required=False,
        help="The identicator of the specific prediction task",
    )

    return parser


def get_serve_parser(subparsers):
    parser = subparsers.add_parser("serve", help="serve a model")
    # parser = subparser
    parser.add_argument("--data_cfg", type=str, required=True, help="The data config, s3 address")
    parser.add_argument(
        "--load_state",
        type=str,
        default=None,
        help="Load a pre-training load_state to continue refining, s3 address",
    )
    parser.add_argument("--task_id", type=str, default="task_default_none")
    return parser
