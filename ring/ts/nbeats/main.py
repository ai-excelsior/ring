import argparse

from ring.common.cmd_parsers import get_predict_parser, get_train_parser, get_validate_parser


def train():
    pass


def validate():
    pass


def predict():
    pass


def serve():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    get_train_parser(subparsers)
    get_validate_parser(subparsers)
    get_predict_parser(subparsers)

    args = parser.parse_args()
    print(args)
