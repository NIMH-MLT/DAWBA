from pathlib import Path
import argparse
from shutil import rmtree


def build_parser():
    parser = argparse.ArgumentParser(
        prog="remove_dir",
    )
    parser.add_argument("--dir_path", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dir_to_delete = Path(args.dir_path)
    print("*" * 50)
    print("*" * 50)
    print("*" * 50)
    for layer_dir in dir_to_delete.iterdir():
        print(f"{layer_dir.name}...")
        if layer_dir.is_file():
            layer_dir.unlink()
            continue
        for agg_dir in layer_dir.iterdir():
            print(f"\t{agg_dir.name}...")
            if agg_dir.is_file():
                agg_dir.unlink()
                continue
            for impact_dir in agg_dir.iterdir():
                print(f"\t\t{impact_dir.name}...")
                if impact_dir.is_file():
                    impact_dir.unlink()
                    continue
                for llama_dir in impact_dir.iterdir():
                    if llama_dir.is_file():
                        llama_dir.unlink()
                        continue
                    for desc_seg_dir in llama_dir.iterdir():
                        if desc_seg_dir.is_file():
                            desc_seg_dir.unlink()
                            continue
                        for alpha_dir in desc_seg_dir.iterdir():
                            if alpha_dir.is_file():
                                alpha_dir.unlink()
                                continue
                            for repeat_dir in alpha_dir.iterdir():
                                if repeat_dir.is_file():
                                    repeat_dir.unlink()
                                    continue
                                rmtree(repeat_dir)
    rmtree(dir_to_delete)


if __name__ == "__main__":
    main()
