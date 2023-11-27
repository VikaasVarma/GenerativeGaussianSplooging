"""
Given a series of directories, evaluates the PSNR/SSIM on all of them
"""
import os
import argparse
import util
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="Root directory")
    parser.add_argument("-gt", "--gt", type=str, help="Ground truth directory")
    args = parser.parse_args()

    f = os.listdir(args.dir)

    d = {}

    for folder in f:
        if not os.path.isfile(os.path.join(args.dir, folder, "0.png")):
            print("Skipping", folder)
            continue
        if os.path.join(args.dir, folder) == args.gt:  # (wont work for relative paths etc but not that deep)
            print("Skipping GT folder:", folder)
            continue
        if not folder.startswith("l2guidedsd_ddim"): continue

        psnr, ssim = util.calculate_psnr_ssim(os.path.join(args.dir, folder), args.gt, quiet=True)
        print(f"PSNR = {psnr:.3f}, SSIM = {ssim:.4f}\t\t{folder}")

        d[folder] = (psnr, ssim)

    pickle.dump(d, open("out/results", "wb"))
