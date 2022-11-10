# tallon.py
#
# The code that defines the behaviour of Tallon. This is the place
# (the only place) where you should write code, using access methods
# from world.py, and using makeMove() to generate the next move.
#
# Written by: Simon Parsons
# Last Modified: 12/01/22

import world
import random
import utils
import config
import numpy as np
import mdptoolbox
from utils import Directions

class Tallon():
    grid_size = (config.worldBreadth, config.worldLength)

    w_c_r = -0.04       # This command is for the reward for moving to a white cell

    b_c_r= 1.0          # This command is for the reward for moving to a bonus cell

    m_c_r= -1.0         # This command is for the reward for moving to a meanie cell

    p_c_r = -1.0        # This command is for the reward for moving to a black hole cell

    other_sides = (1 - config.directionProbability)/2
    action_north_south_east_west_prob=(other_sides, other_sides, config.directionProbability, 0.) # This line of code is to find the probability of moving in the intended direction

    def __init__(self, arena):
        # Make a copy of the world an attribute, so that Tallon can

        # query the state of the world

        self.gameWorld = arena

        # What moves are possible.

        # These are the possible moves of the tallon which are of four direction

        self.moves = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        
    def makeMove(self):
        try:
            self.tallon_location = self.gameWorld.getTallonLocation()     # This command is to get the position of the tallon

            self.bonuses = self.gameWorld.getBonusLocation()        # This command is to get the bonus location
            self.pits = self.gameWorld.getPitsLocation()            # This command is to get the position of pit
            self.meanies = self.gameWorld.getMeanieLocation()        # This command is to get the position of meanies
            
            self.num_states = self.grid_size[0] * self.grid_size[1]
            self.num_actions = 4

            P,R = self.fill_in_probs()

            # check that the reward and probability matrices are well-formed, and match.

            # Success is silent, failure displays error messages

            mdptoolbox.util.check(P, R)
            #run value iteration, discount value is set to 0.99
            vi2 = mdptoolbox.mdp.ValueIteration(P, R, 0.99)
            vi2.run()

             # In this, the Position of the tallon, meanis, pit and the bonus are displayed


            print('The Policy are:\n', vi2.policy)

            tallon_position = lambda x: np.ravel_multi_index(x, self.grid_size)
            tallon_position_in_grid = int(tallon_position((self.tallon_location.y,self.tallon_location.x)))

            print("Tallon's position in grid:",tallon_position_in_grid)

            if int(vi2.policy[tallon_position_in_grid]) == 0:
                print("At the grid position",tallon_position_in_grid," tallon is moving towards", vi2.policy[tallon_position_in_grid],"north")
                return Directions.NORTH
            if int(vi2.policy[tallon_position_in_grid]) == 1:
                print("At the grid position",tallon_position_in_grid," tallon is moving towards",vi2.policy[tallon_position_in_grid],"south")
                return Directions.SOUTH
            if int(vi2.policy[tallon_position_in_grid]) == 2:
                print("At the grid position",tallon_position_in_grid,"tallon is moving towards", vi2.policy[tallon_position_in_grid],"east")
                return Directions.EAST
            if int(vi2.policy[tallon_position_in_grid]) == 3:
                print("At the grid position",tallon_position_in_grid,"tallon is moving towards",vi2.policy[tallon_position_in_grid],"west")
                return Directions.WEST
        except Exception as e:
            print(" Tallon can't locate any bonus in the arena. Moving randomly")
            random_pose = utils.pickRandomPose(self.tallon_location.y,self.tallon_location.x)
            if random_pose.x > self.tallon_location.x:
                return Directions.EAST
            if random_pose.x < self.tallon_location.x:
                return Directions.WEST
            if random_pose.y > self.tallon_location.y:
                return Directions.NORTH
            if random_pose.y < self.tallon_location.y:
                return Directions.SOUTH 

    def fill_in_probs(self):
        try:
            P = np.zeros((self.num_actions, self.num_states, self.num_states))      #create the probability array
            R = np.zeros((self.num_states, self.num_actions)) 

            # This array is used to store all bonus location                   
            a_b=[]  
            # This array is used to store all pits location                                                                 
            a_p=[]  
            # This array is used to store all meanies location                                                            
            a_m=[]                                                           

            tallon_position = lambda x: np.ravel_multi_index(x, (self.grid_size))
            # To determine the Tallons Position in the grid
            tallon_position_in_grid = tallon_position((self.tallon_location.y,self.tallon_location.x))
            # The next three lines of code are used to determine the pit, manies and bonus grid position


            p_p = lambda x: np.ravel_multi_index(x,self.grid_size) # Pit position
            m_p = lambda x: np.ravel_multi_index(x,self.grid_size) # Meanies position
            b_p = lambda x: np.ravel_multi_index(x,self.grid_size) # Bonus position

            
            #https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value

            # This part is to determine the closest bonus to tallon so that it can be printed in the output window:

            
            for bonus in range(len(self.bonuses)):
                a_b.append(b_p((self.bonuses[bonus].y,self.bonuses[bonus].x)))
                # closest bonus to tallon
                c_b = min(a_b,key=lambda x:abs(x-tallon_position_in_grid))
                c_b = str(c_b)
                c_b = (int(c_b[0]),int(c_b[1])) if (len(c_b)>1) else (0,int(c_b))
                currentbonus = c_b
            print("The Bonus that is  closest to Tallon is: ",c_b)

            # This part is to determine the closest meanie to tallon so that it can be printed in the output window:

            for meanie in range(len(self.meanies)):
                a_m.append(m_p((self.meanies[meanie].y,self.meanies[meanie].x)))
                #closest meanie
                c_m = min(a_m,key=lambda x:abs(x-tallon_position_in_grid))
                c_m = str(c_m)
                c_m = (int(c_m[0]),int(c_m[1])) if (len(c_m)>1) else (0,int(c_m))
                currentmeanie = c_m
            print("The Meanie that is closest to Tallon is: ",c_m)

            # This part is to determine the closest pit to tallon so that it can be printed in the output window:
            for pit in range(len(self.pits)):
                a_p.append(p_p((self.pits[pit].y,self.pits[pit].x)))
                #closest pit
                c_p = min(a_p,key=lambda x:abs(x-tallon_position_in_grid))
                c_p = str(c_p)
                c_p = (int(c_p[0]),int(c_p[1])) if(len(c_p)>1) else (0,int(c_p))
                currentpit = c_p
            print("The Pit that is closest to Tallon is: ",c_p)

            #convert grid to 1d for processing    
            grid_to_1d = lambda x: np.ravel_multi_index(x, self.grid_size)

            def hit_wall(cell):
                try:
                    grid_to_1d(cell)
                except ValueError as e:
                    return True
                return False

            # make probs for each action
            North = [self.action_north_south_east_west_prob[i] for i in (0, 1, 2, 3)]
            South = [self.action_north_south_east_west_prob[i] for i in (1, 0, 3, 2)]
            West = [self.action_north_south_east_west_prob[i] for i in (2, 3, 1, 0)]
            East = [self.action_north_south_east_west_prob[i] for i in (3, 2, 0, 1)]
            actions = [North, South, East, West]
            for i, a in enumerate(actions):
                actions[i] = {'North':a[2], 'South':a[3], 'West':a[0], 'East':a[1]}
            
            def update_P_and_R(cell, new_cell, a_index, a_prob):
                if cell == currentmeanie:
                    P[a_index, grid_to_1d(cell), grid_to_1d(cell)] = 1.0
                    R[grid_to_1d(cell), a_index] = self.m_c_r
                
                elif cell == currentpit:
                    P[a_index, grid_to_1d(cell), grid_to_1d(cell)] = 1.0
                    R[grid_to_1d(cell), a_index] = self.p_c_r

                elif cell == currentbonus:
                    P[a_index, grid_to_1d(cell), grid_to_1d(cell)] = 1.0
                    R[grid_to_1d(cell), a_index] = self.b_c_r

                elif hit_wall(new_cell):  # add prob to current cell
                    P[a_index, grid_to_1d(cell), grid_to_1d(cell)] += a_prob
                    R[grid_to_1d(cell), a_index] = self.w_c_r

                else:
                    P[a_index, grid_to_1d(cell), grid_to_1d(new_cell)] = a_prob
                    R[grid_to_1d(cell), a_index] = self.w_c_r

            for a_index, action in enumerate(actions):
                for cell in np.ndindex(self.grid_size):
                    #North
                    new_cell = (cell[0]-1, cell[1])
                    update_P_and_R(cell, new_cell, a_index, action['North'])

                    #South
                    new_cell = (cell[0]+1, cell[1])
                    update_P_and_R(cell, new_cell, a_index, action['South'])

                    #West
                    new_cell = (cell[0], cell[1]-1)
                    update_P_and_R(cell, new_cell, a_index, action['West'])

                    #East
                    new_cell = (cell[0], cell[1]+1)
                    update_P_and_R(cell, new_cell, a_index, action['East'])
            return P,R

        except Exception as e:
            print()