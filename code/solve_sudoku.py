def notsafe(s, x, y, t):
    for i in range(9):
        if s[x][i] == t or s[i][y] == t or s[int(x / 3) * 3 + i % 3][int(y / 3) * 3 + int(i / 3)] == t:
            return 1
    return 0


def solve_sudoku(sudoku, x, y):
    if(x == 0 and y == 9):
        return sudoku
    if(sudoku[x][y] == 0):
        for i in range(1, 10):
            if(notsafe(sudoku, x, y, i)):
                continue
            sudoku[x][y] = i
            if(x == 8):
                if(y == 8):
                    return sudoku
                else:
                    t = solve_sudoku(sudoku, 0, y + 1)
            else:
                t = solve_sudoku(sudoku, x + 1, y)
            if t != 0:
                return sudoku
            sudoku[x][y] = 0
    else:
        if(x == 8):
            t = solve_sudoku(sudoku, 0, y + 1)
        else:
            t = solve_sudoku(sudoku, x + 1, y)
        if t != 0:
            return sudoku
    return 0