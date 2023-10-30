#implement gomoku game
#starting board is zeros
#white is 1, black is -1
import numpy as np

turnRef = {1:"White", -1:"Black"}

'''
input:
	state: board state (15x15 numpy array)
	turn: next move {-1, 1}
output:
	state: an updated board state
'''
def make_move(state, turn):
	validCords = False
	while not validCords:
		coords = input("coords for move, x, y")
		x, y = coords.split(',') 
		x = int(x.strip())
		y = int(y.strip())

		if state[x][y] == 0:
			state[x][y] = turn
			validCords = True
	
	return state


'''
input:
	state: board state (15x15 numpy array)
output:
	win: boolean value of whether current player won
'''
def check_win():
	pass


def play_game(state):
	#check if winner
	#call next turn
	gameIsRunning = True
	curMove = -1 #start turn is always white (flips in loop)
	

	while gameIsRunning:
		curMove *= -1 #switch turns
		state = make_move(state, curMove)
		print(state)
		gameIsRunning = not check_win()
		




if __name__ == "__main__":
	back = np.zeros((15, 15))
	play_game(back)

