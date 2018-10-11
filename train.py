from lib.training import Experiment

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    return args


def read_yaml(path):

    with open(path, "r") as f:
        config = yaml.load(f)

    return config


if __name__ == "__main__":

    args = parse_args()
    config = read_yaml(args.config)

    exper = Experiment(config)
    exper.run()
