from parser_options import ParserOptions
from util.general_functions import write_csv_file
from constants import *
from core.trainers.trainer import Trainer

def main():
    args = ParserOptions().parse()  # get training options
    args.inference = 1

    trainer = Trainer(args)
    image_paths, predictions = trainer.inference()
    write_csv_file(image_paths, predictions)

if __name__ == "__main__":
    main()