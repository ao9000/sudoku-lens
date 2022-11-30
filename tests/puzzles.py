"""
    Benchmarking Sudoku Puzzles
    Puzzles data retrieved from https://github.com/t-dillon/tdoku/blob/master/data.zip
"""

import pytest
BLANK_STATE = 0


def puzzle_conversion(dataset):
    # Data cleaning & conversion
    cleaned_dataset = []
    for line in dataset:
        # Convert empty cells to EMPTY_STATE
        line = line.rstrip().replace(".", str(BLANK_STATE))
        # Split string into our sudoku format
        line = [[int(i) for i in line[x:x + 9]] for x in range(0, len(line), 9)]
        # Record into new list
        cleaned_dataset.append(line)

    return cleaned_dataset


@pytest.fixture()
def puzzle_kaggle():
    # Path to data file
    filename = "tests/data/puzzles0_kaggle"
    with open(filename) as file:
        # Get rid of headers
        dataset = file.readlines()[2:]

    return puzzle_conversion(dataset)


@pytest.fixture()
def puzzle_17clue():
    # Path to data file
    filename = "tests/data/puzzles2_17_clue"
    with open(filename) as file:
        # Get rid of headers
        dataset = file.readlines()[5:]

    return puzzle_conversion(dataset)


@pytest.fixture()
def puzzle_forum_hardest():
    # Path to data file
    filename = "tests/data/puzzles6_forum_hardest_1106"
    with open(filename) as file:
        # Get rid of headers
        dataset = file.readlines()[3:]

    return puzzle_conversion(dataset)


@pytest.fixture()
def puzzle_gen_invalid():
    # Path to data file
    filename = "tests/data/puzzles8_gen_puzzles"
    with open(filename) as file:
        # Get rid of headers
        dataset = file.readlines()[3:]

    return puzzle_conversion(dataset)
