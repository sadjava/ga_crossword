import argparse

from genetic_algorithm import evolution
from utils import plot_crossword


GRID_SIZE = 20

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("words_file", type=str)
    args = parser.parse_args()

    with open(args.words_file, "r") as f:
        words = [w.strip('\n').strip() for w in f.readlines()]
    best, fitness = evolution(words, grid_size=GRID_SIZE, verbose=True)
    print(f"Best crossword (fitness {fitness}):")
    print(*best, sep='\n')
    plot_crossword(best, fitness, grid_size=GRID_SIZE)
    