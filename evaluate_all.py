"""
Given a series of directories, evaluates the PSNR/SSIM on all of them
"""
import os
import argparse
import util
import pickle


# Some utils that allow printing in a sensible order
def key(name):
    xs = name.split("_")
    xs = sum([i.split("=") for i in xs], [])
    ys = []
    for i in xs:
        try: ys.append(float(i))
        except: ys.append(i)
    return tuple(ys)

def to_int(name):
    ints = [i for i in bytes(name, "utf-8")]
    t = 0
    for i in ints:
        t *= 256
        t += i
    return t

def all_keys(names):
    def mp(key):
        key = [to_int(i) if type(i) == str else i for i in key]
        return key + [0] * (mx - len(key))
    keys = [key(i) for i in names]
    mx = max(map(len, keys))
    return {name: mp(i) for name, i in zip(names, keys)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, help="Root directory")
    parser.add_argument("-gt", "--gt", type=str, help="Ground truth directory")
    args = parser.parse_args()

    f = os.listdir(args.dir)
    kmap = all_keys(f)
    f = sorted(f, key=lambda i: kmap[i])

    out_file = "out/results"
    d = {}
    if os.path.isfile(out_file):
        d = pickle.load(open(out_file, "rb"))

    pc = -1
    for folder in f:
        if not os.path.isfile(os.path.join(args.dir, folder, "0.png")):
            print("Skipping", folder)
            continue
        if os.path.join(args.dir, folder) == args.gt:  # (wont work for relative paths etc but not that deep)
            print("Skipping GT folder:", folder)
            continue
        if folder in d:
            psnr, ssim = d[folder]
        else:
            psnr, ssim = util.calculate_psnr_ssim(os.path.join(args.dir, folder), args.gt, quiet=True)

        nc = len(key(folder))
        if nc != pc: print()
        pc = nc
        print(f"PSNR = {psnr:.3f}, SSIM = {ssim:.4f}\t\t{folder}")

        d[folder] = (psnr, ssim)

    pickle.dump(d, open(out_file, "wb"))
