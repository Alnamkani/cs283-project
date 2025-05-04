#!/usr/bin/env python3
import argparse

def filter_lines(input_path: str, threshold: float, full_out: str, seq_out: str) -> None:
    """
    Reads a file where each line is: <sequence> <score>
    Writes to `full_out` only those lines with score >= threshold,
    and to `seq_out` only the sequences (no scores) of those lines.
    """
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(full_out, 'w', encoding='utf-8') as fout_full, \
         open(seq_out, 'w', encoding='utf-8') as fout_seq:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue  # skip malformed lines
            seq, score_str = parts
            try:
                score = float(score_str)
            except ValueError:
                continue  # skip lines where score isn't a float
            if score >= threshold:
                fout_full.write(f"{seq}\t{score_str}\n")
                fout_seq.write(f"{seq}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Filter lines by score and split out sequences"
    )
    parser.add_argument('input_file',
                        help="Path to the input file (each line: sequence<TAB>score)")
    parser.add_argument('threshold', type=float,
                        help="Minimum score to keep a line (e.g. 0.9)")
    parser.add_argument('--output-full', default='filtered_full.txt',
                        help="Output path for full lines that pass the threshold")
    parser.add_argument('--output-seq', default='filtered_seqs.txt',
                        help="Output path for just the sequences of passing lines")
    args = parser.parse_args()

    filter_lines(
        input_path=args.input_file,
        threshold=args.threshold,
        full_out=args.output_full,
        seq_out=args.output_seq
    )

if __name__ == '__main__':
    main()
