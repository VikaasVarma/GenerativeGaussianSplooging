import argparse
import subprocess
import os
import json
import shutil
from copy import deepcopy

import numpy as np
import cv2


def random_transform_matrices(dataset: np.ndarray, num_transforms: int = 30, max_rotation: float = np.pi / 4):
    # Generates random camera poses given distribution of previous poses
    translations = dataset[:, :3, 3]
    new_translations = np.random.normal(
        translations.mean(axis=0),
        translations.std(axis=0),
        size=(num_transforms, 3)
    )

    rotation_matrices = []

    # Pose of default OpenGL camera
    initial_pose = np.array([0, 0, -1])
    for translation in new_translations:
        # Get rotation matrix pointing camera at origin
        final_pose = -translation / np.linalg.norm(translation)
        v = np.cross(initial_pose, final_pose)
        c = np.dot(initial_pose, final_pose)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        
        # Rotate camera randomly around y axis
        random_rotation = (np.random.random() * 2 - 1) * max_rotation
        rotation_matrix = np.array([
            [np.cos(random_rotation), 0, np.sin(random_rotation)],
            [0, 1, 0],
            [-np.sin(random_rotation), 0, np.cos(random_rotation)]
        ]) @ rotation_matrix

        rotation_matrices.append(rotation_matrix)

    return [
        np.vstack((np.hstack((rotation_matrix, translation[:, None])), [0, 0, 0, 1]))
        for rotation_matrix, translation in zip(rotation_matrices, new_translations)
    ]


def generate_transforms(strategy: str, data_dir: str, prev_frames: str, idx, num_frames: int = 10):
    match strategy:
        case "fill":
            # Picks poses farthest from existing poses
            pass
        case "random":
            # Chooses random poses
            transformations = random_transform_matrices(np.array([frame["transform_matrix"] for frame in prev_frames]), num_transforms=num_frames)
        case _:
            raise ValueError(f"Unknown strategy {strategy}")
        
    new_frames = [
        {
            "file_path": f"./train/render_{idx}_{i}",
            "rotation": prev_frames[0]["rotation"],
            "transform_matrix": transformation.tolist(),
        }
        for i, transformation in enumerate(transformations)
    ]

    # Generate blank ground-truth images
    h, w = cv2.imread(os.path.join(data_dir, f"{prev_frames[0]['file_path']}.png")).shape[:2]
    for i in range(num_frames):
        cv2.imwrite(os.path.join(data_dir, f"./train/render_{idx}_{i}.png"), np.zeros((h, w, 3)))

    return new_frames


def train_model(data_dir: str, model_path: str, checkpoint: str | None, iterations: int):

    subprocess.Popen((
        f"python train.py -s {data_dir} " +
        f"-m {model_path} " +
        ("" if checkpoint is None else f"--start_checkpoint {checkpoint} ") +
        f"--iterations {iterations} "
        f"--checkpoint_iterations {iterations}"
    ),
        shell=True
    ).wait()


def apply_diffusion(render_dir: str, out_dir: str):
    # Uses ControlNet image-to-image as this is the best performing method

    # This is a bit ugly but whatever
    repo_root = "../"
    batch_size = 1

    # (probably easiest to use subprocess)
    subprocess.Popen((
        f"python {os.path.join(repo_root, 'cnet_im2im.py')} -s {args.denoise_strength} "
        f"--control-strength {args.control_strength} "
        f"-d {render_dir} "
        f"-o {out_dir} "
        f"-b {batch_size} "
        f"--controlnet {args.controlnet_path} "
        "-m runwayml/stable-diffusion-v1-5 "  # will download on first run
    ),
        shell=True
    ).wait()


def render_samples(data_dir: str, model_path: str, idx: int, iterations: int, strategy: str = "random", num_new_frames: int = 10):
    transforms_file = os.path.join(data_dir, 'transforms_train.json')
    with open(transforms_file, 'r') as f:
        transforms = json.load(f)
    
    new_frames = generate_transforms(strategy, data_dir, transforms["frames"], idx, num_new_frames)
    
    with open(transforms_file, 'w') as f:
        _transforms = deepcopy(transforms)
        _transforms["frames"] = new_frames
        json.dump(_transforms, f)

    subprocess.Popen((
        f"python render.py -s {data_dir} "
        f"-m {model_path} "
        "--skip_test"
    ),
        shell=True
    ).wait()

    render_dir = os.path.join(model_path, "train", f"ours_{iterations}", "renders")
    out_dir = os.path.join(model_path, "train", f"ours_{iterations}", "processed")
    apply_diffusion(render_dir, out_dir)

    for i in range(len(new_frames)):
        shutil.copyfile(os.path.join(out_dir, f"{i:05d}.png"), os.path.join(data_dir, f"./train/render_{idx}_{i}.png"))


def train(data_dir: str, model_path: str, train_iterations: int, retrain_iterations: int, num_new_frames: int):
    # Empty val and test transform files
    with open(os.path.join(data_dir, 'transforms_val.json'), 'w') as f:
        json.dump({"camera_angle_x": 0,  "frames": []}, f)

    with open(os.path.join(data_dir, 'transforms_test.json'), 'w') as f:
        json.dump({"camera_angle_x": 0, "frames": []}, f)

    for i in range(retrain_iterations):
        train_model(
            data_dir,
            os.path.join(model_path, 'model'),
            None if i == 0 else os.path.join(model_path, 'model', f'chkpnt{train_iterations * i}.pth'),
            train_iterations * (i + 1)
        )

        render_samples(data_dir, os.path.join(model_path, 'model'), iterations=train_iterations * (i + 1), idx=i, num_new_frames=num_new_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args")

    parser.add_argument("--data_dir", type=str, default="../dataset", help="input path")
    parser.add_argument("--checkpoint_path", type=str, default="../checkpoints", help="output path")
    parser.add_argument("--train_iterations", type=int, default=3_000, help="number of iterations before retraining")
    parser.add_argument("--retrain_iterations", type=int, default=3, help="number of retraining steps")
    parser.add_argument("--controlnet_path", type=str, required=True, help="ControlNet model path")
    parser.add_argument("--denoise-strength", type=float, default=0.6, help="Denoising strength (0 to 1).")
    parser.add_argument("--control-strength", type=float, default=1.5, help="Control strength. 1-2 seems to give good results.")
    parser.add_argument("--num-new-frames", type=int, default=10, help="Number of new frames to generate per existing frame")

    args = parser.parse_args()
    train(args.data_dir, args.checkpoint_path, args.train_iterations, args.retrain_iterations, args.num_new_frames)
