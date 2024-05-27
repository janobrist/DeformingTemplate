from training.train import training_main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--datasets", type=list, default=["Paper"])
    parser.add_argument("--cameras", type=list, default=["0029", "0068"])
    parser.add_argument("--force_features", type=bool, default=True)

    parser.add_argument("--chamfer_weight", type=float, default=1.0)
    parser.add_argument("--roi_chamfer_weight", type=float, default=0.0)
    parser.add_argument("--normals_weight", type=float, default=0.0)
    parser.add_argument("--roi_normals_weight", type=float, default=0.0)
    args = parser.parse_args()

    training_main(args)