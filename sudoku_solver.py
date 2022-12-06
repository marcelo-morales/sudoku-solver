'''
Simulate an AI solver to solve a Sudoku puzzle.
Reference: https://www.activestate.com/blog/solving-sudoku-with-python-and-artificial-intelligence/ 
'''

import numpy as np
import pandas as pd
import time
import itertools

sudoku_df = pd.DataFrame(pd.read_csv('sudoku.csv', nrows = 10**2))


def shape(sudoku_df):
    for n in range(sudoku_df.shape[0]):
        sudoku_df.iloc[n,0] = np.reshape(list(sudoku_df.puzzle.values[n]),(9,9)).astype(int)
        sudoku_df.iloc[n,1] = np.reshape(list(sudoku_df.solution.values[n]),(9,9)).astype(int)
    return sudoku_df


sudoku_df = shape(sudoku_df)
sudoku_df.iloc[0,0]

def checkPuzzle(sudoku_puzzle):
    checkRow = all([all([x in sudoku_puzzle[nrow,:] for x in range(1,10)]) for nrow in range(9)])
    checkCol = all([all([x in sudoku_puzzle[:,ncol] for x in range(1,10)]) for ncol in range(9)])

    checkUpperLeft = all([x in sudoku_puzzle[0:3,0:3] for x in range(1,10)])
    checkUpperMid = all([x in sudoku_puzzle[0:3,3:6] for x in range(1,10)])
    checkUpperRight = all([x in sudoku_puzzle[0:3,6:9] for x in range(1,10)])

    checkMidLeft = all([x in sudoku_puzzle[3:6,0:3] for x in range(1,10)])
    checkMidMid = all([x in sudoku_puzzle[3:6,3:6] for x in range(1,10)])
    checkMidRight = all([x in sudoku_puzzle[3:6,6:9] for x in range(1,10)])

    checkLowerLeft = all([x in sudoku_puzzle[6:9,0:3] for x in range(1,10)])
    checkLowerMid = all([x in sudoku_puzzle[6:9,3:6] for x in range(1,10)])
    checkLowerRight = all([x in sudoku_puzzle[6:9,6:9] for x in range(1,10)])

    solved = all([checkRow,checkCol,checkUpperLeft,checkUpperMid,checkUpperRight,
                  checkMidLeft,checkMidMid,checkMidRight,checkLowerLeft,checkLowerMid,checkLowerRight])
    if solved:
        for line in sudoku_puzzle:
            print(*line)
    return solved

len(np.where(sudoku_df.iloc[0,0] == 0)[0])
"{:.2e}".format(9**len(np.where(sudoku_df.iloc[0,0] == 0)[0]))

def determineValues(sudoku_puzzle):
    puzzle_values = list()
    for r in range(9):
        for c in range(9):
            if sudoku_puzzle[r,c] == 0:
                cell_values = np.array(range(1,10))
                cell_values = np.setdiff1d(cell_values,sudoku_puzzle[r,:][np.where(sudoku_puzzle[r,:] != 0)]).tolist()
                cell_values = np.setdiff1d(cell_values,sudoku_puzzle[:,c][np.where(sudoku_puzzle[:,c] != 0)]).tolist()
            else:
                cell_values = [sudoku_puzzle[r,c]]
            puzzle_values.append(cell_values)
    return puzzle_values

puzzle_values = determineValues(sudoku_df.iloc[0,0])

combinations = 1
for i in range(81):
    combinations = combinations*len(puzzle_values[i])
"{:.2e}".format(combinations) 

'1.25e+19'

def checkPuzzle(sudoku_puzzle):
    checkRow = all([all([x in sudoku_puzzle[nrow,:] for x in range(1,10)]) for nrow in range(9)])
    checkCol = all([all([x in sudoku_puzzle[:,ncol] for x in range(1,10)]) for ncol in range(9)])

    checkUpperLeft = all([x in sudoku_puzzle[0:3,0:3] for x in range(1,10)])
    checkUpperMid = all([x in sudoku_puzzle[0:3,3:6] for x in range(1,10)])
    checkUpperRight = all([x in sudoku_puzzle[0:3,6:9] for x in range(1,10)])

    checkMidLeft = all([x in sudoku_puzzle[3:6,0:3] for x in range(1,10)])
    checkMidMid = all([x in sudoku_puzzle[3:6,3:6] for x in range(1,10)])
    checkMidRight = all([x in sudoku_puzzle[3:6,6:9] for x in range(1,10)])

    checkLowerLeft = all([x in sudoku_puzzle[6:9,0:3] for x in range(1,10)])
    checkLowerMid = all([x in sudoku_puzzle[6:9,3:6] for x in range(1,10)])
    checkLowerRight = all([x in sudoku_puzzle[6:9,6:9] for x in range(1,10)])

    solved = all([checkRow,checkCol,checkUpperLeft,checkUpperMid,checkUpperRight,
                  checkMidLeft,checkMidMid,checkMidRight,checkLowerLeft,checkLowerMid,checkLowerRight])
    if solved:
        for line in sudoku_puzzle:
            print(*line)
    return solved

def determineValues(sudoku_puzzle):
    puzzle_values = list()
    for r in range(9):
        for c in range(9):
            if sudoku_puzzle[r,c] == 0:
                cell_values = np.array(range(1,10))
                cell_values = np.setdiff1d(cell_values,sudoku_puzzle[r,:][np.where(sudoku_puzzle[r,:] != 0)]).tolist()
                cell_values = np.setdiff1d(cell_values,sudoku_puzzle[:,c][np.where(sudoku_puzzle[:,c] != 0)]).tolist()
            else:
                cell_values = [sudoku_puzzle[r,c]]
            puzzle_values.append(cell_values)
    return puzzle_values

def bruteForce_check(puzzle_values):
    first = np.array(np.meshgrid(*puzzle_values[0:27])).T.reshape(-1,3,9)
    second = np.array(np.meshgrid(*puzzle_values[27:54])).T.reshape(-1,3,9)
    third = np.array(np.meshgrid(*puzzle_values[54:])).T.reshape(-1,3,9)

    start_time = time.time()
    for i in range(first.shape[0]):
        for j in range(second.shape[0]):
            for k in range(third.shape[0]):
                potential_solution = np.concatenate((first[i],second[j], third[k]))
                solution = checkPuzzle(potential_solution)
                iterations = 10**4
                if (i+1)*(j+1)*(k+1) == iterations:
                    current_time = time.time()
                    run_time = current_time - start_time
                    print('Projected Number of Days: ')
                    print("{:.2e}".format(combinations*(run_time/iterations)/(24*3600)))
                    break
                else:
                    continue
            break
        break

def checkGrids(r,c,sudoku_puzzle,n):
    if r < 3:
        if c < 3:
            subgrid = n in sudoku_puzzle[0:3,0:3]
        elif c < 6:
            subgrid = n in sudoku_puzzle[0:3,3:6] 
        else:
            subgrid = n in sudoku_puzzle[0:3,6:9]
    elif r < 6:
        if c < 3:
            subgrid = n in sudoku_puzzle[3:6,0:3]
        elif c < 6:
            subgrid = n in sudoku_puzzle[3:6,3:6]
        else:
            subgrid = n in sudoku_puzzle[3:6,6:9]        
    else:
        if c < 3:
            subgrid = n in sudoku_puzzle[6:9,0:3]
        elif c < 6:
            subgrid = n in sudoku_puzzle[6:9,3:6]
        else:
            subgrid = n in sudoku_puzzle[6:9,6:9]  
    return subgrid

def solve(sudoku_puzzle,puzzle_values):
    count = 0
    solution = False
    rows = np.array(np.where(sudoku_puzzle == 0))[0]
    cols = np.array(np.where(sudoku_puzzle == 0))[1]
    dic = dict(zip(list(range(len(rows))), np.zeros(len(rows),dtype = int).tolist()))
    while solution == False:
        if count >= len(rows):
            solution = checkPuzzle(sudoku_puzzle)
            break
        r = rows[count]
        c = cols[count]
        len_num = len(np.array(puzzle_values).reshape(9,9)[r,c])
        num = dic[count]
        while num < len_num:
            cell = np.array(puzzle_values).reshape(9,9)[r,c][num]
            checkRow = cell in sudoku_puzzle[r,:]
            if checkRow:
                num += 1
                continue
            checkCol = cell in sudoku_puzzle[:,c]
            if checkCol:
                num += 1
                continue
            checkGrid = checkGrids(r,c,sudoku_puzzle,cell)
            if checkGrid:
                num += 1
                continue
            dic[count] = num
            count += 1
            sudoku_puzzle[r,c] = cell
            break
        else:
            sudoku_puzzle[r,c] = 0 
            dic[count] = 0
            count -= 1 
    



