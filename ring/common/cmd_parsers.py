def get_train_parser(subparsers):
    parser = subparsers.add_parser("train", help="train a model")
    # parser = subparser
    parser.add_argument("--train_data", type=str, required=True, help="The train data source, s3 address")
    parser.add_argument("--val_data", type=str, required=True, help="The val data source, s3 address")
    parser.add_argument(
        "--model_state",
        type=str,
        default=None,
        help="The base model state, useful when refine a model, s3 address",
    )

    # common trainer config
    parser.add_argument("--loss", type=str, default="MSELoss", help="The loss function")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_clip_grad_norm", type=float, default=None)

    return parser


def get_validate_parser(subparsers):
    parser = subparsers.add_parser("validate", help="validate a model, get validation metrics")
    parser.add_argument("--data", type=str, required=True, help="The val data source, s3 address")
    parser.add_argument(
        "--model_state",
        type=str,
        required=True,
        help="The base model state, useful when refine a model, s3 address",
    )

    return parser


def get_predict_parser(subparsers):
    parser = subparsers.add_parser("predict", help="predict the last part of the given data")
    parser.add_argument("--data", type=str, required=True, help="The data source, s3 address")
    parser.add_argument(
        "--model_state",
        type=str,
        required=True,
        help="The base model state, useful when refine a model, s3 address",
    )

    return parser
