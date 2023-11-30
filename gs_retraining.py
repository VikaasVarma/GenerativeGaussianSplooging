import argparse
import subprocess
import os
import json
from copy import deepcopy

import numpy as np


def random_tranform_matrices(dataset: np.ndarray, num_transforms: int = 30, max_rotation: float = np.pi / 4):
    # Generates random camera poses given distribution of previous poses
    translations = dataset[:, :3, 3]
    new_translations = np.random.normal(
        translations.mean(axis=0),
        translations.std(axis=0),
        size=(num_transforms, 3)
    )

    rotations_matrices = []

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
        random_rotation = (np.random() * 2 - 1) * max_rotation
        rotation_matrix = np.array([
            [np.cos(random_rotation), 0, np.sin(random_rotation)],
            [0, 1, 0],
            [-np.sin(random_rotation), 0, np.cos(random_rotation)]
        ]) @ rotation_matrix

        rotations_matrices.append(rotation_matrix)

    return [
        np.vstack((np.hstack((rotation_matrix, rotation_matrix @ translation[:, None])), [0, 0, 0, 1]))
        for rotation_matrix, translation in zip(rotations_matrices, new_translations)
    ]


def generate_transforms(strategy: str, prev_frames: str, idx, num_frames: int = 30):
    match strategy:
        case "fill":
            # Picks poses farthest from existing poses
            pass
        case "random":
            # Chooses random poses
            transformations = random_tranform_matrices(np.array([frame["transform"] for frame in prev_frames]), num_transforms=num_frames)
        case _:
            raise ValueError(f"Unknown strategy {strategy}")
        
    new_frames = [
        {
            "file_path": f"render_{idx}_{i}",
            "rotation": prev_frames[0]["rotation"],
            "transformation_matrix": transformation.tolist(),
        }
        for i, transformation in enumerate(transformations)
    ]
    return new_frames


def train_model(data_dir: str, model_path: str, checkpoint: str | None, iterations: int):

    subprocess.Popen((
        f"python train.py -s {data_dir} "
        f"-m {model_path} "
        "" if checkpoint is None else f"--start_checkpoint {checkpoint}"
        f"--iterations {iterations}"
    ),
        shell=True
    ).wait()


def apply_diffusion(samples: list[str]):    
    for sample in samples:
        # TODO: Apply diffusion to sample
        pass

    # TODO: return filepaths of new images
    return []


def render_samples(data_dir: str, model_path: str, idx: int, strategy: str = "random"):
    transforms_file = os.path.join(data_dir, 'transforms_train.json')
    with open(transforms_file, 'r') as f:
        transforms = json.load(f)
    
    new_frames = generate_transforms(strategy, transforms["prev_frames"], idx)
    
    with open(transforms_file, 'w') as f:
        _transforms = deepcopy(transforms)
        _transforms["frames"] = new_frames
        json.dump(_transforms, f)

    subprocess.Popen((
        f"python render.py -s {transforms_file} "
        f"-m {model_path}"
    ),
        shell=True
    ).wait()

    diffused_samples = apply_diffusion(data_dir, [frame["file_path"] for frame in new_frames])

    with open(transforms_file, 'w') as f:
        for frame, sample in zip(new_frames, diffused_samples):
            frame["file_path"] = sample

        transforms["frames"] += new_frames
        json.dump(transforms, f)

    return [frame["file_path"] for frame in new_frames]

def train(data_dir: str, model_path: str, train_iterations: int, retrain_iterations: int):
    for i in range(retrain_iterations):
        train_model(
            data_dir,
            os.path.join(model_path, f"checkpoint_{i}.pth"),
            None if i == 0 else os.path.join(model_path, f"checkpoint_{i - 1}.pth"),
            train_iterations
        )

        render_samples(data_dir, model_path, idx = i)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args")

    parser.add_argument("--data_dir", type=str, default="../dataset", help="input path")
    parser.add_argument("--checkpoint_path", type=str, default="../checkpoints", help="output path")
    parser.add_argument("--train_iterations", type=int, default=3_000, help="number of iterations before retraining")
    parser.add_argument("--retrain_iterations", type=int, default=3, help="number of retraining steps")

    args = parser.parse_args()
    train(args.data_dir, args.checkpoint_path, args.train_iterations, args.retrain_iterations)
