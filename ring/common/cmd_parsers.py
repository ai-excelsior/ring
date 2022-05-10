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
        help="Load a pre-training model_state to continue refining, s3 address",
    )
    parser.add_argument(
        "--save_state",
        type=str,
        default=None,
        help="Upload the training model_state for validation, prediction or later refining, oss address",
    )

    # common trainer config
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_clip_grad_norm", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--train_gaussian_percentage", type=float, default=0.25)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--logger_mode", type=str, default="local")
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
        required=True,
        help="The measurement name when saving predictions result to influxdb",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        required=True,
        help="The identicator of the specific prediction task",
    )

    return parser
