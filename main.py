import argparse

def main(args):


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse terminal arguments for functions.")
    
    # Add a positional argument for the function name
    parser.add_argument("base model", type=str, choices=["mlp", "lgbm"])

    # Add arguments for the 'some_function'
    parser.add_argument("--val_split", type=float, help="Argument 1 for 'some_function'")
    parser.add_argument("--test_split", type=float, help="Argument 2 for 'some_function'")

    # Add argument for the 'another_function'
    parser.add_argument("--arg3", type=str, help="Argument for 'another_function'")

    # Parse the arguments
    args = parser.parse_args()
    
    # Call main with the parsed arguments
    main(args)