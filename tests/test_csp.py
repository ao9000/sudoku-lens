from tests.puzzles import puzzle_kaggle, puzzle_17clue, puzzle_forum_hardest, puzzle_gen_invalid
from constraint_satisfaction_problem import csp
import time
from statistics import fmean
BLANK_STATE = 0

TEST_SIZE = 3


def test_puzzle_gen_invalid(puzzle_gen_invalid):
    timings = []

    for puzzle in puzzle_gen_invalid[:TEST_SIZE]:
        start_time = time.time()
        solved, step = csp(puzzle)
        time_taken = time.time() - start_time

        print(f"Time taken: {time_taken:.6f}")
        timings.append(time_taken)

        assert step == 0
        assert solved == puzzle

    print(f"\nMean time taken: {fmean(timings):.6f}")


def test_puzzle_kaggle(puzzle_kaggle):
    timings = []

    for puzzle in puzzle_kaggle[:TEST_SIZE]:
        start_time = time.time()
        solved, step = csp(puzzle)
        time_taken = time.time() - start_time

        print(f"Time taken: {time_taken:.6f}")
        timings.append(time_taken)

        assert step != 0
        assert not any(BLANK_STATE in row for row in solved)

    print(f"\nMean time taken: {fmean(timings):.6f}")


def test_puzzle_17clue(puzzle_17clue):
    timings = []

    for puzzle in puzzle_17clue[:TEST_SIZE]:
        start_time = time.time()
        solved, step = csp(puzzle)
        time_taken = time.time() - start_time

        print(f"Time taken: {time_taken:.6f}")
        timings.append(time_taken)

        assert step != 0
        assert not any(BLANK_STATE in row for row in solved)

    print(f"\nMean time taken: {fmean(timings):.6f}")


def test_puzzle_forum_hardest(puzzle_forum_hardest):
    timings = []

    for puzzle in puzzle_forum_hardest[:TEST_SIZE]:
        start_time = time.time()
        solved, step = csp(puzzle)
        time_taken = time.time() - start_time

        print(f"Time taken: {time_taken:.6f}")
        timings.append(time_taken)

        assert step != 0
        assert not any(BLANK_STATE in row for row in solved)

    print(f"\nMean time taken: {fmean(timings):.6f}")