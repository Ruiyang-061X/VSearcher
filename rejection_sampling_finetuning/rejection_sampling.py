import argparse

from rejection_sampling_finetuning.rft_util import *


def rejection_sampling(trajectory_list):
    rejection_sampled_trajectory_list = []
    for trajectory in trajectory_list:
        if trajectory["correct"]:
            rejection_sampled_trajectory_list.append(trajectory)
    return rejection_sampled_trajectory_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory_jsonl_path", type=str, required=True)
    parser.add_argument("--output_jsonl_path", type=str, required=True)
    
    args = parser.parse_args()

    trajectory_list = load_jsonl(args.trajectory_jsonl_path)
    rejection_sampled_trajectory_list = rejection_sampling(trajectory_list)
    save_jsonl(rejection_sampled_trajectory_list, args.output_jsonl_path)