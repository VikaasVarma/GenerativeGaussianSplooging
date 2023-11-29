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
        psnrs, ssims = d[folder]
    else:
        psnrs, ssims = util.calculate_psnr_ssim(os.path.join(args.dir, folder), args.gt, quiet=True, return_list=True)
        d[folder] = (psnrs, ssims)
    return psnrs, ssims


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

    psnrs_i, ssims_i = query("input")
    psnrs_i = np.array(psnrs_i)
    ssims_i = np.array(ssims_i)
    ord_p = np.argsort(psnrs_i)
    ord_s = np.argsort(ssims_i)

    for folder in f:
        if not os.path.isfile(os.path.join(args.dir, folder, "0.png")):
            print("Skipping", folder)
            continue
        if os.path.join(args.dir, folder) == args.gt:  # (wont work for relative paths etc but not that deep)
            print("Skipping GT folder:", folder)
            continue
        import random
        if random.uniform(0, 1) < 0.8: continue

        psnrs, ssims = query(folder)

        if len(psnrs) == len(psnrs_i):
            ord_p2 = np.argsort(np.array(psnrs))
            ord_s2 = np.argsort(np.array(ssims))
            xs = np.arange(len(psnrs))

            plt.scatter(xs, psnrs_i[ord_p], marker='o')
            plt.scatter(xs, np.array(psnrs)[ord_p], marker='x')
            plt.title(folder)
            plt.ylabel("PSNR")
            plt.show()

            # plt.scatter(xs, ssims_i[ord_s], marker='o')
            # plt.scatter(xs, np.array(ssims)[ord_s], marker='x')
            # plt.title(folder)
            # plt.ylabel("SSIM")
            # plt.show()
            import time
            time.sleep(1.0)

    pickle.dump(d, open(out_file, "wb"))
