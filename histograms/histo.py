#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def plot_line_length_distribution(input_path: str, output_path: str = None, bin_width: int = 5) -> None:
    """
    Reads each line from input_path, measures its length (excluding the newline),
    and plots a histogram of those lengths with specified bin width, alternating
    bin colors, formatted y-axis ticks, and summary statistics annotated.
    """
    # Read line lengths
    lengths = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            lengths.append(len(line.rstrip('\n')))

    total_lines = len(lengths)
    if total_lines == 0:
        print("No lines to plot.")
        return

    # Compute statistics
    min_len = int(np.min(lengths))
    max_len = int(np.max(lengths))
    mean_len = np.mean(lengths)
    median_len = np.median(lengths)

    # Define bins
    bins = np.arange(0, max_len + bin_width, bin_width)

    # Plot histogram
    fig, ax = plt.subplots()
    counts, edges, patches = ax.hist(lengths, bins=bins, edgecolor='black')

    # Alternate bin colors
    colors = ['lightgrey', 'grey']
    for i, patch in enumerate(patches):
        patch.set_facecolor(colors[i % 2])

    # Format y-axis ticks for readability (commas for thousands)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

    # Labels and title
    ax.set_xlabel('Peptide length (characters)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Peptide Lengths')

    # Annotate summary stats in the top-left corner
    stats_text = (
        f"Total peptides: {total_lines:,}\n"
        f"Min length: {min_len}\n"
        f"Max length: {max_len}\n"
        f"Mean length: {mean_len:.2f}\n"
        f"Median length: {median_len:.0f}"
    )
    ax.text(
        0.05, 0.95, stats_text,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
    )

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Saved histogram to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot the distribution of peptide lengths"
    )
    parser.add_argument('input_file', help="Path to the input text file")
    parser.add_argument(
        '-o', '--output',
        help="Path to save the histogram image (e.g. histogram.png). If omitted, the plot will be shown interactively."
    )
    parser.add_argument(
        '-b', '--bin-width',
        type=int,
        default=5,
        help="Width of each histogram bin in characters (default: 5)"
    )
    args = parser.parse_args()
    plot_line_length_distribution(args.input_file, args.output, args.bin_width)

if __name__ == '__main__':
    main()
