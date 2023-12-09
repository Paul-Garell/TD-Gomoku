import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
from IPython.display import display
import pandas as pd


class gomoku_game:
    def __init__(self):
        self.dim = 15 #15x15 board, can be 19x19
        self.state = np.zeros([self.dim, self.dim], dtype=np.int8)
        self.players = {'p1': 1, 'p2': 2} 
        self.isDone = False
        self.reward = {'win': 1, 'draw': 0.5, 'loss': -1}
        self.available = [] #initialize to all spots on board
        for i in range(self.dim):
            for j in range(self.dim):
                self.available.append(i + j* self.dim)
    
    def boardDim(self):
        return self.dim
    
    def gameIsDone(self):
        return self.isDone

    def render(self): #rev
        rendered_board_state = self.state.copy().astype(str)
        rendered_board_state[self.state == 0] = ' '
        rendered_board_state[self.state == 1] = 'B'
        rendered_board_state[self.state == 2] = 'W'
        display(pd.DataFrame(rendered_board_state))
    
    def reset(self): #rev
        self.__init__()
        
    def get_available_actions(self):
        return self.available
    
    def check_game_done(self, player, last):
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
                right = ri - 1
            else:
                right = ri
            if right - left >= 4: 
                return True 
            return False
        def check_diag(state, i, j):
            li = i
            lj = j
            while li > 0 and lj > 0 and state[li][lj] == state[i][j]:
                li -= 1
                lj -= 1
            if state[li][lj] != state[i][j]:
                lowi = li + 1
            else:
                lowi = li
            ui = i
            uj = j
            while ui < len(state) - 1 and uj < len(state) - 1 and state[ui][uj] == state[i][j]:
                ui += 1
                uj += 1
            if state[ui][uj] != state[i][j]:
                upi = ui - 1
            else:
                upi = ui

            if upi - lowi >= 4: 
                return True
            return False

        i, j = last
        vert = check_flat(self.state, i, j)  #I dont believe these need to be copied
        horiz = check_flat(self.state.T, j, i)
        diagl = check_diag(self.state, i, j)
        diagr = check_diag(np.fliplr(self.state), i, len(self.state) -1 - j)
        diag = diagl or diagr
        return (horiz or vert) or diag

    def make_move(self, a, player):
        # check if move is valid
        i = a % self.dim
        j = int((a - i)/ self.dim)
        openSpots = self.get_available_actions()
        if a in openSpots:
            self.state[i,j] = self.players[player]
            self.available.remove(a)
        else:
            print('Move is invalid')
            self.render()
        a = (i, j)
        win = self.check_game_done(player, a)

        reward = 0.
        if len(openSpots) == 0:
            reward = self.reward['draw']
            self.isDone = True
        elif win:  
            reward = self.reward['win']
            self.isDone = True
        return self.state, reward

class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        # 6 by 7, 10 by 11 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(7,7), padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(7,7), padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(7,7), padding=2)
        # self.conv4 = nn.Conv2d(128, 128, kernel_size=4, padding=2)
        # self.conv5 = nn.Conv2d(128, 64, kernel_size=4, padding=2)
        # self.conv6 = nn.Conv2d(64, 32, kernel_size=4, padding=2)
        # self.conv7 = nn.Conv2d(32, 32, kernel_size=4, padding=2)

        # linear_input_size = 6 * 7 * 32
        self.MLP1 = nn.Linear(2592, 1024)
        self.MLP2 = nn.Linear(1024, 1024)
        # self.MLP3 = nn.Linear(50, 50)
        self.MLP4 = nn.Linear(1024, outputs)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        # x = F.leaky_relu(self.conv4(x))
        # x = F.leaky_relu(self.conv5(x))
        # x = F.leaky_relu(self.conv6(x))
        # x = F.leaky_relu(self.conv7(x))
        # flatten the feature vector except batch dimension
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.MLP1(x))
        x = F.leaky_relu(self.MLP2(x))
        # x = F.leaky_relu(self.MLP3(x))
        return self.MLP4(x)


def select__action_inference(state, available_actions, neuralnet):
    with torch.no_grad():
        all_probabilities = neuralnet(torch.tensor(state, dtype=torch.float, device=device).unsqueeze(dim=0).unsqueeze(dim=0))[0, :] #unsqez
        potential_action_probs = [all_probabilities[a] for a in available_actions]
        return available_actions[np.argmax(potential_action_probs)] 




'''
make inference takes in the current state, next player, and the the neural network. 
It outputs the best next move to be played. 
env should be of type GomokuGame
player should be one of {'p1', 'p2'}
and neural net should be a trained nn of the DQN class

the output is the update board state after a move is done, the move that was taken, and whether the game is done: 
The gameIsDone output is {-1, 0, 1} corresponding to -1= game is not done, 0=game is draw, 1=current player won

The output correpsonding to the move is in the format "r-c" as a string
'''
def make_inference(env, player, neural_net):
    available_actions = env.get_available_actions()
    action = select__action_inference(env.state, available_actions, neural_net)
    state, reward = env.make_move(action, player)

    gameIsDone = -1
    if reward == 1:
        gameIsDone = 1
    elif env.gameIsDone() == False:
        gameIsDone = 0

    row = action % env.boardDim()
    action = action - row
    col = action / env.boardDim()
    stringify = str(row) + "-" + str(col)

    return state, stringify, gameIsDone

import time 
if __name__ == "__main__": 
    players = ['p1','p2']
    env = gomoku_game()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # net = DQN()
    # optimizer = optim.Adam(net.parameters())
    PATH = "src/savedModels/gomo_200epoch_random_OP.pth"

    # Load
    model = DQN(env.dim **2)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    gameIsRunning = True

    while gameIsRunning:
        state, stringify, gameIsDone = make_inference(env, 'p1', model)
        print(state)
        print(stringify)
        print(gameIsDone)
        time.sleep(5)
        state, stringify, gameIsDone = make_inference(env, 'p2', model)
        print(state)
        print(stringify)
        print(gameIsDone)
        time.sleep(5)
        