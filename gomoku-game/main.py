#implement gomoku game
#starting board is zeros
#white is 1, black is -1
import numpy as np

turnRef = {1:"White", -1:"Black"}
moves = []
# movenum = 0

def fileToList(textFile):
	f = open(textFile, "r")
	L = f.readline()
	while L != '' and L != '\n':
		x, y = L.split(',') 
		x = int(x.strip())
		y = int(y.strip())
		moves.append((x,y))
		L = f.readline()
	return moves

'''
return x,y coordinates
'''
def getInput(cli, movenum):
	if cli:
		coords = input("coords for move, x, y")
		x, y = coords.split(',') 
		x = int(x.strip())
		y = int(y.strip())
		return x, y
	else:
		# L = getFrom list
		return moves[movenum]



'''
input:
	state: board state (15x15 numpy array)
	turn: next move {-1, 1}
	inputs: True if CLI interface, false for text file interface
output:
	state: an updated board state
'''
def make_move(state, turn, inputs, movenum):
	inputs
	validCords = False
	while not validCords:
		x,y = getInput(inputs, movenum)
		# x, y = coords.split(',') 
		# x = int(x.strip())
		# y = int(y.strip())

		if state[x][y] == 0:
			state[x][y] = turn
			validCords = True
	
	return state, (x, y)

'''
vary in i direction
keep j constant 
'''
def check_flat(state, i, j):
	ci = i
	while ci > 0 and state[ci][j] == state[i][j]: 
		ci -= 1
	if state[ci][j] != state[i][j]:
		left = ci + 1
	else:
		left = ci
	ri = i
	while ri < (len(state) - 1) and state[ri][j] == state[i][j]: 
		ri += 1 
	if state[ri][j] != state[i][j]:
		right = ri + 1
	else:
		right = ri
	if ri - ci >= 5: 
		return True 
	return False

'''
find length of current diag with equal values and if over 4 return true 
'''
def check_diag(state, i, j):
	li = i
	lj = j
	while li > 0 and lj > 0 and state[li][lj] == state[i][j]:
		li -= 1
		lj -= 1
	if state[li][lj] != state[i][j]:
		lowi = li + 1
		lowj = lj + 1
	else:
		lowi = li
		lowj = lj

	ui = i
	uj = j
	while ui < len(state) - 1 and uj < len(state) - 1 and state[ui][uj] == state[i][j]:
		ui += 1
		uj += 1
	if state[ui][uj] != state[i][j]:
		upi = ui + 1
		upj = uj + 1
	else:
		upi = ui
		upj = uj

	if ui - li >= 5: 
		return True
	return False

'''
input:
	state: board state (15x15 numpy array)
output:
	win: boolean value of whether current player won
'''
def check_win(state, last):
	#check diagonal
	#check horizontal
	#check vertical
	i, j = last
	vert = check_flat(state, i, j)
	horiz = check_flat(state.T, j, i)
	
	# print("hor" + str(horiz))
	# print("vert" + str(vert))
	
	#i think this should work??
	diagl = check_diag(state, i, j)
	diagr = check_diag(np.fliplr(state), i, len(state) -1 - j)

	diag = diagl or diagr

	# print("diag "+str(diag))
	return (horiz or vert) or diag
	

def play_game(state, inputs):
	#check if winner
	#call next turn
	gameIsRunning = True
	curMove = -1 #start turn is always white (flips in loop)
	movenum = 0

	while gameIsRunning:
		print("\n MOVE NUMBER: " + str(movenum))
		curMove *= -1 #switch turns
		state, move = make_move(state, curMove, inputs, movenum)
		movenum += 1
		print(state)
		gameIsRunning = not check_win(state,move)
		if np.sum(np.sum(np.abs(state[:, :]))) == 225:
			gameIsRunning = False
		
	if check_win(state, move):
		print(str(turnRef[curMove]) + " won!")
	else:
		print("board full, no winner.")

	

if __name__ == "__main__":
	cli = False
	textFile = "test/input.txt"
	if not cli:
		fileToList(textFile)
	back = np.zeros((15, 15))
	play_game(back, cli)

