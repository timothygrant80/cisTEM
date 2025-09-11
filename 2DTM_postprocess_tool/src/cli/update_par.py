#!/usr/bin/env python3

import argparse
import pandas as pd
from io import StringIO
from tm_post import starfile

def read_par_file(par_path):
    """Read a cisTEM .par file into header, data DataFrame, and footer."""
    with open(par_path, "r") as f:
        lines = f.readlines()

    header = lines[0]
    footer = lines[-2:]
    data_lines = lines[1:-2]

    data_str = ''.join(data_lines)
    df = pd.read_csv(StringIO(data_str), delim_whitespace=True, header=None)
    df.columns = header.strip().split()

    return header, df, footer


def read_score_file(score_path):
    """Read a file with one score per line."""
    df = starfile.load_particle_starfile(score_path)
    return df["SCORE"].tolist()


def update_scores(df, score_files):
    """Concatenate scores from all files and assign to the SCORE column."""
    all_scores = []
    for path in score_files:
        scores = read_score_file(path)
        all_scores.extend(scores)

    if len(all_scores) != len(df):
        raise ValueError(f"Number of scores ({len(all_scores)}) does not match number of particles ({len(df)}).")

    df['SCORE'] = all_scores
    return df


def write_par_file(out_path, header, df, footer):
    """Write the updated .par file."""
    with open(out_path, "w") as f:
        f.write(header)
        for row in df.itertuples(index=False):
            values = '   '.join(f"{v:>8}" if isinstance(v, float) else f"{v:>8}" for v in row)
            f.write(f"{values}\n")
        f.writelines(footer)


def main():
    parser = argparse.ArgumentParser(description="Update SCORE column in a cisTEM .par file.")
    parser.add_argument("par_file", help="Path to the original .par file")
    parser.add_argument("score_files", nargs='+', help="One or more star files containing updated SCORE values")
    parser.add_argument("-o", "--output", required=True, help="Output path for updated .par file")

    args = parser.parse_args()

    header, df, footer = read_par_file(args.par_file)
    df = update_scores(df, args.score_files)
    write_par_file(args.output, header, df, footer)

    print(f"[INFO] Updated .par file written to: {args.output}")


if __name__ == "__main__":
    main()