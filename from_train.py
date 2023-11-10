"""
Utility script for producing a noisy dataset by training to a small number of epochs
"""
import argparse
import os
import subprocess
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train_dir", type=str, required=True, help="Directory containing scenes to train with")
parser.add_argument("-r", "--render_dir", type=str, required=True, help="Directory containing scenes to render")
parser.add_argument("-s", "--scenes", nargs="+", type=str, required=True, help="Which scenes in the data directory to use")
parser.add_argument("-i", "--iters", nargs="+", type=int, required=True, help="Iterations to sample")
parser.add_argument("-o", "--out", type=str, required=True, help="Location to store the resulting dataset")
parser.add_argument("-mo", "--model_out", type=str, required=True, help="Location to save the models")
parser.add_argument("-ta", "--train_args", type=str, default="", help="Extra args for training")
parser.add_argument("-ra", "--render_args", type=str, default="", help="Extra args for rendering")
parser.add_argument("--skip_train", action="store_true", help="Render from a pre-trained model without retraining")
parser.set_defaults(skip_train=False)


def train_cmd(data_path, model_out, iters, train_args):
    max_iters = max(iters)
    iters = " ".join(map(str, sorted(iters)))
    s = f"python train.py -s {data_path} -m {model_out} --iterations {max_iters} --save_iterations {iters}"
    if train_args != "":
        s = s + " " + train_args
    return s



def render_cmds(data_path, model_out, iters, render_args):
    for i in iters:
        yield f"python render.py -m {model_out} --iteration {i} -s {data_path}" + " " + render_args


if __name__ == "__main__":
    args = parser.parse_args()

    for p in os.listdir(args.train_dir):
        if p not in args.scenes:
            continue
        print()
        print("*"*20)
        print(f"Processing scene '{p}'")
        print()

        data_path = os.path.join(args.train_dir, p)  # contains test/train data
        model_path = os.path.join(args.model_out, p)
        render_path = os.path.join(args.render_dir, p)

        os.makedirs(model_path, exist_ok=True)

        # Train
        if not args.skip_train:
            cmd = train_cmd(data_path, model_path, args.iters, args.train_args)
            print(f"Executing '{cmd}'")
            code = subprocess.call(cmd, shell=True)
            if code != 0:
                print("ABORTING: Exit code", code)
                quit(code)

            print()
            print("*"*20)
            print("Training completed, beginning renders")
            print()

        for cmd in render_cmds(render_path, model_path, args.iters, args.render_args):
            print(f"Executing '{cmd}'")
            code = subprocess.call(cmd, shell=True)
            if code != 0:
                print("WARNING: Exit code was", code)

        print("\nRendering completed, moving files...")

        for i in args.iters:
            print(f" Moving from iter {i}...")
            base_path = os.path.join(args.out, p, f"render_{i}")
            train_path = os.path.join(base_path, "train")
            test_path = os.path.join(base_path, "test")

            os.makedirs(base_path, exist_ok=True)

            out_train_path = os.path.join(args.model_out, p, "train", f"ours_{i}", "renders")
            out_test_path = os.path.join(args.model_out, p, "test", f"ours_{i}", "renders")

            shutil.move(out_train_path, train_path)
            shutil.move(out_test_path, test_path)

            # Move GT too
            if i == args.iters[0]:
                print(" Moving gt...")
                base_path = os.path.join(args.out, p, "gt")
                train_path = os.path.join(base_path, "train")
                test_path = os.path.join(base_path, "test")

                os.makedirs(base_path, exist_ok=True)

                out_train_path = os.path.join(args.model_out, p, "train", f"ours_{i}", "gt")
                out_test_path = os.path.join(args.model_out, p, "test", f"ours_{i}", "gt")

                shutil.move(out_train_path, train_path)
                shutil.move(out_test_path, test_path)
            # Delete GT output as it is duplicated
            else:
                gt1 = os.path.join(args.model_out, p, "train", f"ours_{i}", "gt")
                gt2 = os.path.join(args.model_out, p, "test", f"ours_{i}", "gt")
                shutil.rmtree(gt1)
                shutil.rmtree(gt2)

        print("Moving completed.")
