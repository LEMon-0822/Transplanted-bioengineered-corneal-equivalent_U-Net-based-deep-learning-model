# from train import *
from test import *

parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=0.001, type=float, dest="lr")
parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=200 , type=int, dest="num_epoch")


parser.add_argument("--data_dir", default="D:/Collagen sheet_Datasets/Training datasets_Doctor3", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="D:/Collagen sheet_Results_Doctor3/result_AttU-Net/Noise_checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="D:/Collagen sheet_Results_Doctor3/result_AttU-Net/log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="D:/Collagen sheet_Results_Doctor3/result_AttU-Net/result", type=str, dest="result_dir")

parser.add_argument("--task", default="super_resolution", choices=["None", "denoising", "inpainting", "super_resolution"], type=str, dest="task")
parser.add_argument("--opts", nargs='+', default=["bilinear", 10, 1], dest="opts")      ## SRResnet은 down sampling이 진행되기 때문에 keepdim을 off 해야함
parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")

parser.add_argument("--ny", default=512, type=int, dest="ny")
parser.add_argument("--nx", default=512, type=int, dest="nx")
parser.add_argument("--nch", default=1, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")
parser.add_argument("--out_channels", default=2, type=int, dest="out_channels")

parser.add_argument("--network", default="resnet", choices=["unet", "resnet", "hourglass", "srresnet", "resnet"], type=str, dest="network")
parser.add_argument("--learning_type", default="residual", choices=["plane", "residual"],type=str, dest="learning_type")

parser.add_argument("--train_continue", default="on", type=str, dest="train_continue")
args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
