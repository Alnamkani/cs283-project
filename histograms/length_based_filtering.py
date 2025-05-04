#!/usr/bin/env python3
import argparse

def filter_lines(input_path: str, output_path: str, max_length: int = 24) -> None:
    """
    Reads each line from input_path and writes it to output_path
    only if its length (excluding the newline) is less than max_length.
    """
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            text = line.rstrip('\n')
            if len(text) < max_length:
                fout.write(line)

def main():
    parser = argparse.ArgumentParser(
        description="Keep only lines shorter than a given length"
    )
    parser.add_argument('input_file', help="Path to the input text file")
    parser.add_argument(
        '-m', '--max-length',
        type=int,
        default=24,
        help="Maximum line length (default: 24 characters)"
    )
    args = parser.parse_args()
    filter_lines(args.input_file, "filtered_"+args.input_file, args.max_length)

if __name__ == '__main__':
    main()
