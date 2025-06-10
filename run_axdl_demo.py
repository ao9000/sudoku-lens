import numpy as np
import cv2
from algorithm_x_dancing_links import algorithm_x_dl, BLANK_STATE
from copy import deepcopy


def draw_puzzle(board):
    # Draw board
    # First, create a black mat image
    board_image = np.zeros((600, 600), np.uint8)

    # Second, draw sudoku lines
    for num in range(1, 9):
        if num % 3:
            # Horizontal
            cv2.line(board_image, (0, (600 // 9) * num), (600, (600 // 9) * num), (255, 255, 255), 2)
            # Vertical
            cv2.line(board_image, ((600 // 9) * num, 0), ((600 // 9) * num, 600), (255, 255, 255), 2)
        else:
            # Every 3rd line draw line with more thickness
            # Horizontal
            cv2.line(board_image, (0, (600 // 9) * num), (600, (600 // 9) * num), (255, 255, 255), 5)
            # Vertical
            cv2.line(board_image, ((600 // 9) * num, 0), ((600 // 9) * num, 600), (255, 255, 255), 5)

    # Third, draw number in board
    for row in range(0, 9):
        for col in range(0, 9):
            # Check if cell is not a blank state (0)
            if board[row][col] != BLANK_STATE:
                cv2.putText(board_image, str(board[row][col]),
                            ((600 // 9) * col + (600 // 27), (600 // 9) * row + (600 // 14)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2)

    return board_image


def main():
    # Sample of invalid board
    # board = np.array([
    #     [5, 9, 0, 0, 7, 0, 0, 0, 0],
    #     [6, 0, 0, 1, 3, 5, 0, 0, 0],
    #     [0, 7, 8, 0, 0, 0, 0, 6, 0],
    #     [8, 0, 0, 0, 6, 0, 0, 0, 3],
    #     [4, 0, 0, 8, 0, 3, 0, 0, 1],
    #     [7, 0, 0, 0, 2, 0, 0, 0, 6],
    #     [0, 6, 0, 0, 0, 0, 2, 8, 0],
    #     [0, 0, 0, 4, 1, 9, 0, 0, 5],
    #     [0, 0, 0, 0, 8, 0, 0, 7, 9],
    # ])

    # # Sample of invalid board, starting invalid, key error
    # board = np.array([
    #     [5, 9, 5, 0, 7, 0, 0, 0, 0],
    #     [6, 0, 0, 1, 3, 5, 0, 0, 0],
    #     [0, 7, 8, 0, 0, 0, 0, 6, 0],
    #     [8, 0, 0, 0, 6, 0, 0, 0, 3],
    #     [4, 0, 0, 8, 0, 3, 0, 0, 1],
    #     [7, 0, 0, 0, 2, 0, 0, 0, 6],
    #     [0, 6, 0, 0, 0, 0, 2, 8, 0],
    #     [0, 0, 0, 4, 1, 9, 0, 0, 5],
    #     [0, 0, 0, 0, 8, 0, 0, 7, 9],
    # ])

    # Valid board example
    board = np.zeros((9, 9), dtype=int)

    board[0] = [0, 0, 0, 2, 6, 0, 7, 0, 1]
    board[1] = [6, 8, 0, 0, 7, 0, 0, 9, 0]
    board[2] = [1, 9, 0, 0, 0, 4, 5, 0, 0]
    board[3] = [8, 2, 0, 1, 0, 0, 0, 4, 0]
    board[4] = [0, 0, 4, 6, 0, 2, 9, 0, 0]
    board[5] = [0, 5, 0, 0, 0, 3, 0, 2, 8]
    board[6] = [0, 0, 9, 3, 0, 0, 0, 7, 4]
    board[7] = [0, 4, 0, 0, 5, 0, 0, 3, 6]
    board[8] = [7, 0, 3, 0, 1, 8, 0, 0, 0]

    generator = algorithm_x_dl((3, 3), deepcopy(board))

    # Show unsolved puzzle
    cv2.imshow("Unsolved", draw_puzzle(board))
    cv2.waitKey(0)


    # Cases handling for invalid board
    try:
        solved_board = next(generator, None)
    except KeyError:
        print("Invalid Sudoku board configuration to begin with.")
        solved_board = None

    if solved_board is None:
        print("Invalid puzzle")
    else:
        print("Solution found:")
        print(solved_board)
        # Show solved puzzle
        cv2.imshow(f"Solved", draw_puzzle(solved_board))
        cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()