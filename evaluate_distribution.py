"""
Given a series of directories, evaluates the PSNR/SSIM on all of them
"""
import os
import argparse
import util
import pickle
from evaluate_all import all_keys
import matplotlib.pyplot as plt
import numpy as np


def query(folder):
    if folder in d:
        out = d[folder]
    else:
        out = util.calculate_psnr_ssim(os.path.join(args.dir, folder), args.gt, IND, quiet=True, return_list=True)
        d[folder] = out
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="Root directory")
    parser.add_argument("-gt", "--gt", type=str, help="Ground truth directory")
    args = parser.parse_args()

    f = os.listdir(args.dir)
    kmap = all_keys(f)
    f = sorted(f, key=lambda i: kmap[i])

    out_file = "out/results_dist"
    d = {}
    if os.path.isfile(out_file):
        d = pickle.load(open(out_file, "rb"))

    IND = 3
    names = ["PSNR", "SSIM", "LPIPS", "MS-SSIM"]

    out_i = query("input")
    ord_i = np.argsort(out_i)

    fid = util.calculate_fid(output_dir=os.path.join(args.dir, "input"),
                             gt_dir=args.gt)
    print("input", "\t:", fid)

    for folder in f:
        if not os.path.isfile(os.path.join(args.dir, folder, "0.png")):
            print("Skipping", folder)
            continue
        if os.path.join(args.dir, folder) == args.gt:  # (wont work for relative paths etc but not that deep)
            print("Skipping GT folder:", folder)
            continue
        import random
        if not folder.startswith("cnet_im2im"): continue

        f = util.calculate_fid(output_dir=os.path.join(args.dir, folder),
                               gt_dir=args.gt)
        print(folder, "\t:", f)
        continue

        out = query(folder)

        if len(out) == len(out_i):
            ord2 = np.argsort(np.array(out))
            xs = np.arange(len(out))

            plt.scatter(xs, out_i[ord_i], marker='o')
            plt.scatter(xs, np.array(out)[ord_i], marker='x')
            plt.title(folder)
            plt.ylabel(names[IND])
            plt.show()

            import time
            time.sleep(1.0)

    pickle.dump(d, open(out_file, "wb"))
