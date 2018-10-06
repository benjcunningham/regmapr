from lib.training import Experiment

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    exper = Experiment(args.config)
    exper.run()
