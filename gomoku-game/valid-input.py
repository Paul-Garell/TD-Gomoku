import numpy as np

def validInput(text, size):
    f = open(text, "r")
    L = f.readline()
    board = np.zeros((size, size))
    move = 1
    while L != '' and L != '\n':
        print(L)
        x, y = L.split(',') 
        x = int(x.strip())
        y = int(y.strip())
        if board[x][y] != 0:
            return False
        else:
            board[x][y] = move
        
        move += 1
        L = f.readline()
    return True


if __name__ == "__main__":
    text = "test/input.txt"
    boardSize = 15
    print(validInput(text, boardSize))