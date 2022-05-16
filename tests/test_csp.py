from tests.puzzles import puzzle_kaggle, puzzle_17clue, puzzle_forum_hardest, puzzle_gen_invalid
from csp import csp
import time
from statistics import fmean
BLANK_STATE = 0

TEST_SIZE = 3


def test_puzzle_gen_invalid(puzzle_gen_invalid):
    timings = []
    print()

    for puzzle_index, puzzle in enumerate(puzzle_gen_invalid[:TEST_SIZE], start=1):
        start_time = time.time()
        solved, step = csp(puzzle)
        time_taken = time.time() - start_time

        assert step == 0
        assert solved == puzzle

        print(f"Puzzle {puzzle_index}, time taken: {time_taken:.6f}")
        timings.append(time_taken)

    print(f"Mean time taken: {fmean(timings):.6f}")


def test_puzzle_kaggle(puzzle_kaggle):
    timings = []
    print()

    for puzzle_index, puzzle in enumerate(puzzle_kaggle[:TEST_SIZE], start=1):
        start_time = time.time()
        solved, step = csp(puzzle)
        time_taken = time.time() - start_time

        assert step != 0
        assert not any(BLANK_STATE in row for row in solved)

        print(f"Puzzle {puzzle_index}, time taken: {time_taken:.6f}")
        timings.append(time_taken)

    print(f"Mean time taken: {fmean(timings):.6f}")


def test_puzzle_17clue(puzzle_17clue):
    timings = []
    print()

    for puzzle_index, puzzle in enumerate(puzzle_17clue[:TEST_SIZE], start=1):
        start_time = time.time()
        solved, step = csp(puzzle)
        time_taken = time.time() - start_time

        assert step != 0
        assert not any(BLANK_STATE in row for row in solved)

        print(f"Puzzle {puzzle_index}, time taken: {time_taken:.6f}")
        timings.append(time_taken)

    print(f"Mean time taken: {fmean(timings):.6f}")


def test_puzzle_forum_hardest(puzzle_forum_hardest):
    timings = []
    print()

    for puzzle_index, puzzle in enumerate(puzzle_forum_hardest[:TEST_SIZE], start=1):
        start_time = time.time()
        solved, step = csp(puzzle)
        time_taken = time.time() - start_time

        assert step != 0
        assert not any(BLANK_STATE in row for row in solved)

        print(f"Puzzle {puzzle_index}, time taken: {time_taken:.6f}")
        timings.append(time_taken)

    print(f"Mean time taken: {fmean(timings):.6f}")
