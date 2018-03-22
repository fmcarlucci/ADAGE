from argparse import ArgumentParser

def get_args():
    args = ArgumentParser()
    args.add_argument("model_path")
    args.add_argument("input_path")
    args.add_argument("output_path")
    return args


if __name__ == "__main__":
    pass