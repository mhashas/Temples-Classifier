from parser_options import ParserOptions
from util.general_functions import print_training_info
from constants import *
from core.trainers.trainer import Trainer

def main():
    args = ParserOptions().parse()  # get training options
    trainer = Trainer(args)

    print_training_info(args)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):

        if args.trainval:
            trainer.run_epoch(epoch, split=TRAINVAL)
        else:
            trainer.run_epoch(epoch, split=TRAIN)

            if epoch % args.eval_interval == (args.eval_interval - 1):
                trainer.run_epoch(epoch, split=VAL)

    if not args.trainval:
        trainer.run_epoch(trainer.args.epochs, split=VAL)
        trainer.summary.writer.add_scalar('val/best_acc', trainer.best_acc, args.epochs)

    trainer.summary.writer.close()
    trainer.save_network()


if __name__ == "__main__":
    main()