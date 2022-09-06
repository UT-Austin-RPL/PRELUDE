import numpy as np
import h5py
import pickle
from pathlib import Path
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
            "--folder",
            default="./save/dataset_sim/corl2022/",
            type=Path,
            )
    parser.add_argument(
            "--demo_path",
            default="./save/dataset_sim/demo_corl2022.hdf5",
            type=Path,
            )

    return parser.parse_args()

def main():
    args = parse_args()

    demo_file_name = str(args.demo_path)
    demo_hdf5_file = h5py.File(demo_file_name, "w")
    print(demo_file_name)

    total = 0
    grp = demo_hdf5_file.create_group("data")
    for (demo_idx, pickle_file) in enumerate(args.folder.glob(f'*/*.pickle')):
       with open(pickle_file, "rb") as f:
           data = pickle.load(f)
       print(data.keys())
       ep_grp = grp.create_group(f"demo_{demo_idx}")
       obs_grp = ep_grp.create_group(f"obs")

       obs_grp.create_dataset("agentview_rgb", data=((np.array(data["observation"])[..., :3]).astype('uint8')))
       obs_grp.create_dataset("agentview_depth", data=np.array(data["observation"])[..., 3:])
       obs_grp.create_dataset("yaw", data=np.expand_dims(data["yaw"], axis=-1))
       ep_grp.create_dataset("actions", data=np.array(data["action"]))
       ep_grp.create_dataset("dones", data=np.array(data["done"]))
       ep_grp.create_dataset("rewards", data=np.array(data["done"]))
       ep_grp.attrs["num_samples"] = len(data["action"])
       total += len(data["action"])

    grp.attrs["total"] = total
    metadata = ""
    grp.attrs["env_args"] = metadata

    demo_hdf5_file.close()

if __name__ == "__main__":
    main()

