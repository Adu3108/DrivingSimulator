from importlib.resources import path
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import time
import pygame, sys
from pygame.locals import *
import random
import math
import argparse

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10

control_actions = {
    'steer' : {
        0 : -3,
        1 : 0,
        2 : 3,
    },
    'acceleration' : {
        0 : -5,
        1 : -3.95,
        2 : 0,
        3 : 3.95,
        4 : 5,
    }
}

def radians_to_degrees(r):
    degrees = 180*r/math.pi
    return degrees

def degrees_to_radians(d):
    radians = d*math.pi/180 
    return radians

def features(state):
    alpha = state[3]

    d = math.sqrt((350-state[0])**2 + (state[1])**2)
    feature1 = 1-d/(700*math.sqrt(2))

    feature2 = 0
    beta = radians_to_degrees(math.acos((350-state[0])/d))
    if (state[0]>250 and (state[1]<-100 or state[1]>100)):
        beta = 90
    if state[1]<=0:
        if alpha>180:
            feature2 = 1-np.abs(360-alpha+beta)/360
        else:
            feature2 = 1-np.abs(alpha-beta)/360
    else:
        if alpha<180:
            feature2 = 1-np.abs(alpha+beta)/360
        else:
            feature2 = 1-np.abs(360-alpha-beta)/360

    left = (350+state[0])/700
    if (state[0]>300 and (state[1]<-100 or state[1]>100)):
        left = (350-state[0])/350
    feature3 = left

    vertical = 1-np.abs(state[1])/350
    if(state[1]>=100):
        vertical = (350-state[1])/700
    elif state[1]<=-100:
        vertical = (350+state[1])/700
    feature4 = vertical

    return np.array([feature1,feature2,feature3,feature4])

def transition(state, action):
    next_state = np.copy(state)
    s = 0 
    if state[2] + control_actions["acceleration"][action[1]] >= 0:
        s = next_state[2] + control_actions["acceleration"][action[1]]/2
        next_state[2] += control_actions["acceleration"][action[1]]
    next_state[3] += control_actions["steer"][action[0]] 
    next_state[3] = next_state[3] % 360
    next_state[1] += s*math.sin(degrees_to_radians(next_state[3]))
    next_state[0] += s*math.cos(degrees_to_radians(next_state[3]))
    return next_state

def distance(x_car,y_car,x_mud,y_mud):
    return math.sqrt((x_car-x_mud)**2 + (y_car-y_mud)**2)

def angles(x_car,y_car,x_mud,y_mud):
    d = 0
    if x_mud>=x_car and y_mud>=y_car:
        d = radians_to_degrees(math.atan(np.abs(y_mud-y_car)/np.abs(x_mud-x_car)))
    elif x_mud<x_car and y_mud>=y_car:
        d = 90 + radians_to_degrees(math.atan(np.abs(x_mud-x_car)/np.abs(y_mud-y_car)))
    elif x_car>=x_mud and y_mud<y_car:
        d = 180 + radians_to_degrees(math.atan(np.abs(y_mud-y_car)/np.abs(x_mud-x_car)))
    elif x_mud>x_car and y_car>y_mud:
        d = 270 + radians_to_degrees(math.atan(np.abs(x_mud-x_car)/np.abs(y_mud-y_car)))
    return d

def closest_mud_pit(x_car,y_car,mud_pits):
    d1 = distance(x_car,y_car,mud_pits[0][0],mud_pits[0][1]) 
    d2 = distance(x_car,y_car,mud_pits[1][0],mud_pits[1][1]) 
    d3 = distance(x_car,y_car,mud_pits[2][0],mud_pits[2][1]) 
    d4 = distance(x_car,y_car,mud_pits[3][0],mud_pits[3][1])
    d = [d1,d2,d3,d4]
    return mud_pits[np.argmin(d)] 

def closest_point(x_car,y_car,x_mud,y_mud):
    points = [[x_mud+50,y_mud+50],[x_mud+50,y_mud-50],[x_mud-50,y_mud+50],[x_mud-50,y_mud-50]]
    d1 = distance(x_car,y_car,x_mud+50,y_mud+50) 
    d2 = distance(x_car,y_car,x_mud+50,y_mud-50) 
    d3 = distance(x_car,y_car,x_mud-50,y_mud+50) 
    d4 = distance(x_car,y_car,x_mud-50,y_mud-50)
    d = [d1,d2,d3,d4]
    return points[np.argmin(d)] 

def mud_features(x_car,y_car,alpha,x_closest,y_closest,x_mud,y_mud):

    feature1 = 1
    feature2 = 1

    # Top Left Corner
    if x_car < x_mud - 50 and y_car < y_mud - 50:
        angle1 = angles(x_car,y_car,x_mud+50,y_mud-50)
        angle2 = angles(x_car,y_car,x_mud-50,y_mud+50) 
        mud_angle = angles(x_car,y_car,x_mud-50,y_mud-50)
        if alpha<=angle1 and distance(x_car,y_car,x_mud-50,y_mud-50)<50:
            feature1 = np.abs(angle1-alpha)/90
            feature2 = distance(x_car,y_car,x_mud-50,y_mud-50)/50
        elif alpha>angle1 and alpha<mud_angle and distance(x_car,y_car,x_mud-50,y_mud-50)<50:
            feature1 = 1-(alpha-angle1)/90
            feature2 = distance(x_car,y_car,x_mud-50,y_mud-50)/50
        elif alpha>=mud_angle and alpha<angle2 and distance(x_car,y_car,x_mud-50,y_mud-50)<50:
            feature1 = 1-np.abs(alpha-angle2)/90
            feature2 = distance(x_car,y_car,x_mud-50,y_mud-50)/50
        elif alpha>=angle2 and alpha<90 and distance(x_car,y_car,x_mud-50,y_mud-50)<50:
            feature1 = np.abs(alpha-angle2)/90
            feature2 = distance(x_car,y_car,x_mud-50,y_mud-50)/50
        elif distance(x_car,y_car,x_mud-50,y_mud-50)<50:
            feature2 = (y_mud-50-y_car)/50

    # Left Side
    elif x_car < x_mud-50 and y_car < y_mud+50 and y_car > y_mud-50:
        angle1 = angles(x_car,y_car,x_mud-50,y_mud+50)
        angle2 = angles(x_car,y_car,x_mud-50,y_mud-50)
        if alpha<=angle1 and x_mud-50-x_car<75:
            feature1 = np.abs(alpha)/angle1
            feature2 = (x_mud-50-x_car)/75
        elif alpha>angle1 and alpha<90 and x_mud-50-x_car<75:
            feature1 = 0.95+np.abs(alpha-angle1)/90
            feature2 = distance(x_car,y_car,x_mud-50,y_mud-50)/50
        elif alpha>270 and alpha<=angle2 and x_mud-50-x_car<75:
            feature1 = 0.75+np.abs(alpha-angle2)/360
            feature2 = distance(x_car,y_car,x_mud-50,y_mud+50)/50
        elif alpha>angle2 and x_mud-50-x_car<75:
            feature1 = 1-np.abs(alpha-angle2)/90
            feature2 = (x_mud-50-x_car)/75

    # Bottom Left Corner
    elif x_car<x_mud-50 and y_car>y_mud+50:
        angle1 = angles(x_car,y_car,x_mud-50,y_mud-50)
        angle2 = angles(x_car,y_car,x_mud+50,y_mud+50)
        mud_angle = angles(x_car,y_car,x_mud-50,y_mud+50)
        if alpha>270 and alpha<=angle1 and distance(x_car,y_car,x_mud-50,y_mud+50)<50:
            feature1 = np.abs(angle1-alpha)/90
            feature2 = distance(x_car,y_car,x_mud-50,y_mud+50)/50
        elif alpha>angle1 and alpha<mud_angle and distance(x_car,y_car,x_mud-50,y_mud+50)<50: 
            feature1 = 1-np.abs(alpha-angle1)/90
            feature2 = distance(x_car,y_car,x_mud-50,y_mud+50)/50
        elif alpha>=mud_angle and alpha<angle2 and distance(x_car,y_car,x_mud-50,y_mud+50)<50:
            feature1 = 1-np.abs(alpha-angle2)/90
            feature2 = distance(x_car,y_car,x_mud-50,y_mud+50)/50
        elif alpha>=angle2 and distance(x_car,y_car,x_mud-50,y_mud+50)<50:
            feature1 = np.abs(alpha-angle2)/90
            feature2 = distance(x_car,y_car,x_mud-50,y_mud+50)/50
        elif distance(x_car,y_car,x_mud-50,y_mud+50)<50:
            feature2 = (y_car-y_mud-50)/50

    # Bottom Side
    elif x_car>x_mud-50 and x_car<x_mud+50 and y_car>y_mud+50:
        angle1 = angles(x_car,y_car,x_mud-50,y_mud+50)
        angle2 = angles(x_car,y_car,x_mud+50,y_mud+50)
        if alpha>180 and alpha<=angle1 and y_car-50-y_mud<75:
            feature1 = 0.75+np.abs(alpha-angle1)/360
            feature2 = distance(x_car,y_car,x_mud+50,y_mud+50)/(75*math.sqrt(5))
        elif alpha>angle1 and alpha<270 and y_car-50-y_mud<75:
            feature1 = 1-np.abs(alpha-angle1)/90
            feature2 = (y_car-y_mud-50)/75
        elif alpha>=270 and alpha<angle2 and y_car-50-y_mud<75:
            feature1 = 1-np.abs(alpha-angle2)/90
            feature2 = (y_car-y_mud-50)/75
        elif alpha>=angle2 and y_car-50-y_mud<75:
            feature1 = 0.75+np.abs(alpha-angle2)/360
            feature2 = distance(x_car,y_car,x_mud-50,y_mud+50)/(75*math.sqrt(5))

    # Bottom Right Corner
    elif x_car>x_mud+50 and y_car > y_mud+50:
        angle1 = angles(x_car,y_car,x_mud-50,y_mud+50)
        angle2 = angles(x_car,y_car,x_mud+50,y_mud-50)
        mud_angle = angles(x_car,y_car,x_mud+50,y_mud+50)
        if alpha>180 and alpha<=angle1 and distance(x_car,y_car,x_mud+50,y_mud+50)<50:
            feature1 = np.abs(alpha-angle1)/90
        elif alpha>angle1 and alpha<mud_angle and distance(x_car,y_car,x_mud+50,y_mud+50)<50:
            feature1 = 1-np.abs(alpha-angle1)/90
        elif alpha>=mud_angle and alpha<angle2 and distance(x_car,y_car,x_mud+50,y_mud+50)<50:
            feature1 = 1-np.abs(alpha-angle2)/90
        elif alpha>=angle2 and alpha<270 and distance(x_car,y_car,x_mud+50,y_mud+50)<50:
            feature1 = np.abs(alpha-angle2)/90
        elif distance(x_car,y_car,x_mud+50,y_mud+50)<50:
            feature2 = (y_mud+50-y_car)/50

    # Right Side
    elif x_car>x_mud+50 and y_car > y_mud-50 and y_car<y_mud+50:
        angle1 = angles(x_car,y_car,x_mud+50,y_mud+50)
        angle2 = angles(x_car,y_car,x_mud+50,y_mud-50)
        if alpha>90 and alpha<=angle1 and (x_car-x_mud-50)<75:
            feature1 = 0.75+np.abs(alpha-angle1)/np.abs(angle1-90)
            feature2 = distance(x_car,y_car,x_mud+50,y_mud-50)/50
        elif alpha>angle1 and alpha<180 and (x_car-x_mud-50)<75:
            feature1 = 1-np.abs(alpha-angle1)/np.abs(180-angle1)
            feature2 = (x_car-x_mud-50)/75
        elif alpha>=180 and alpha<angle2 and (x_car-x_mud-50)<75:
            feature1 = 1-np.abs(alpha-angle2)/np.abs(angle2-180)
            feature2 = (x_car-x_mud-50)/75
        elif alpha>=angle2 and alpha<270 and (x_car-x_mud-50)<75:
            feature1 = 0.75+np.abs(alpha-angle2)/np.abs(angle2-270)
            feature2 = distance(x_car,y_car,x_mud+50,y_mud+50)/50

    # Top Right Corner
    elif x_car>x_mud+50 and y_car<y_mud-50:
        angle1 = angles(x_car,y_car,x_mud+50,y_mud+50)
        angle2 = angles(x_car,y_car,x_mud-50,y_mud-50)
        mud_angle = angles(x_car,y_car,x_mud+50,y_mud-50)
        if alpha>90 and alpha<=angle1 and distance(x_car,y_car,x_mud+50,y_mud-50)<50:
            feature1 = np.abs(alpha-angle1)/90
        elif alpha>angle1 and alpha<mud_angle and distance(x_car,y_car,x_mud+50,y_mud-50)<50:
            feature1 = np.abs(alpha-mud_angle)/np.abs(angle2-mud_angle)
        elif alpha>=mud_angle and alpha<angle2 and distance(x_car,y_car,x_mud+50,y_mud-50)<50:
            feature1 = np.abs(alpha-mud_angle)/np.abs(angle2-mud_angle)
        elif alpha>=angle2 and alpha<180 and distance(x_car,y_car,x_mud+50,y_mud-50)<50:
            feature1 = np.abs(alpha-angle2)/90
        elif distance(x_car,y_car,x_mud+50,y_mud-50)<50:
            feature2 = (x_car-x_mud-50)/50

    # Top Side
    elif x_car>x_mud-50 and x_car<x_mud+50 and y_car<y_mud-50:
        angle1 = angles(x_car,y_car,x_mud+50,y_mud-50)
        angle2 = angles(x_car,y_car,x_mud-50,y_mud-50)
        if alpha<=angle1 and (y_mud-50-y_car < 50):
            feature1 = 0.75+np.abs(angle1-alpha)/360
            feature2 = distance(x_car,y_car,x_mud-50,y_mud-50)/(75*math.sqrt(5))
        elif alpha>angle1 and alpha<90 and (y_mud-50-y_car < 50):
            feature1 = 1-np.abs(angle1-alpha)/90
            feature2 = (y_mud-50-y_car)/50
        elif alpha>=90 and alpha<angle2 and (y_mud-50-y_car < 50):
            feature1 = 1-np.abs(angle2-alpha)/90
            feature2 = (y_mud-50-y_car)/50
        elif alpha>=angle2 and alpha<180 and (y_mud-50-y_car < 50):
            feature1 = 0.75+np.abs(alpha-angle2)/360
            feature2 = distance(x_car,y_car,x_mud+50,y_mud-50)/(75*math.sqrt(5))

    return feature1,feature2

def task2_features(state, mud_pits):
    alpha = state[3]

    d = math.sqrt((350-state[0])**2 + (state[1])**2)
    feature1 = 1-d/(700*math.sqrt(2))

    feature2 = 0
    beta = radians_to_degrees(math.acos((350-state[0])/d))
    if (state[0]>250 and (state[1]<-100 or state[1]>100)):
        beta = 90
    if state[1]<=0:
        if alpha>180:
            feature2 = 1-np.abs(360-alpha+beta)/360
        else:
            feature2 = 1-np.abs(alpha-beta)/360
    else:
        if alpha<180:
            feature2 = 1-np.abs(alpha+beta)/360
        else:
            feature2 = 1-np.abs(360-alpha-beta)/360
    
    left = (350+state[0])/700
    if (state[0]>280 and (state[1]<-100 or state[1]>100)):
        left = (350-state[0])/70
    feature3 = left

    vertical = 1-np.abs(state[1])/350
    if(state[1]>=100):
        vertical = (350-state[1])/250
    elif state[1]<=-100:
        vertical = (350+state[1])/250
    feature4 = vertical

    mud_distance = 0
    mud_angle = 0
    if state[1]<-50 or state[1]>50:
        closest = closest_mud_pit(state[0],state[1],mud_pits)
        closest_mud_point = closest_point(state[0],state[1],closest[0],closest[1])
        mud_angle, mud_distance = mud_features(state[0],state[1],state[3],closest_mud_point[0],closest_mud_point[1],closest[0],closest[1])
    
    return np.array([feature1, feature2, feature3, feature4, mud_angle, mud_distance])

def theta_update(theta):
    for i in range(len(theta)):
        if theta[i]<=0:
            theta[i]+=100
    return theta

class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """
        self.epsilon = 0.2
        self.theta = np.array([100.0,100.0,100.0,100])
        self.previous_state = 0
        self.previous_action = 0
        super().__init__()

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        # Replace with your implementation to determine actions to be taken
        
        action_steer = 0
        action_acc = 0
        # Epsilon-Greedy Policy
        if random.random() < self.epsilon:
            action_steer = random.choice([0,1,2])
            action_acc = random.choice([0,1,2,3,4])
        else:
            Q = np.zeros((3,5))
            for i in range(3):
                for j in range(5):
                    Q[i][j] = np.dot(self.theta,features(transition(state,[i,j])))
            #         print(features(transition(state,[i,j])))
            # print(Q)
            actions = np.unravel_index(Q.argmax(), Q.shape)
            action_steer = actions[0]
            action_acc = actions[1]
        action = np.array([action_steer, action_acc])  

        return action

    def controller_task1(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
    
        ######### Do NOT modify these lines ##########
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

        time.sleep(3)
        ##############################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
        
            ######### Do NOT modify these lines ##########
            
            # To keep track of the number of timesteps per epoch
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################

            self.previous_state = state # S^t
            self.previous_action = np.array([1, 2]) # A^t

            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):

                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                cur_time += 1

                state, reward, terminate, reached_road, info_dict = simulator._step(self.previous_action) # S^(t+1), R^t 
                fpsClock.tick(FPS)
                if terminate:
                    self.theta += (reward - np.dot(self.theta,features(transition(self.previous_state, self.previous_action))))*features(transition(self.previous_state, self.previous_action))*0.01
                    road_status = reached_road
                    break
                action = self.next_action(state)
                self.theta += (reward + np.dot(self.theta,features(transition(state, action))) - np.dot(self.theta,features(transition(self.previous_state, self.previous_action))))*features(transition(self.previous_state, self.previous_action))*0.01
                self.previous_state = state
                self.previous_action = action
                theta_update(self.theta)
                
            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))

class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """
        self.epsilon = 0.3
        self.theta = np.array([100.0,100.0,100.0,100.0,200.0,100.0])
        self.previous_state = 0
        self.previous_action = 0
        self.mud_pits = 0
        super().__init__()

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """

        # Replace with your implementation to determine actions to be taken
        action_steer = 0
        action_acc = 0
        # Epsilon-Greedy Policy
        if random.random() < self.epsilon:
            action_steer = random.choice([0,1,2])
            action_acc = random.choice([0,1,2,3,4])
        else:
            Q = np.zeros((3,5))
            for i in range(3):
                for j in range(5):
                    Q[i][j] = np.dot(self.theta,task2_features(transition(state,[i,j]),self.mud_pits))
            actions = np.unravel_index(Q.argmax(), Q.shape)
            action_steer = actions[0]
            action_acc = actions[1]

        action = np.array([action_steer, action_acc])  

        return action
        

    def controller_task2(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        
        ################ Do NOT modify these lines ################
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        time.sleep(3)
        ###########################################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):

            ################ Setting up the environment, do NOT modify these lines ################
            # To randomly initialize centers of the traps within a determined range
            ran_cen_1x = random.randint(120, 230)
            ran_cen_1y = random.randint(120, 230)
            ran_cen_1 = [ran_cen_1x, ran_cen_1y]

            ran_cen_2x = random.randint(120, 230)
            ran_cen_2y = random.randint(-230, -120)
            ran_cen_2 = [ran_cen_2x, ran_cen_2y]

            ran_cen_3x = random.randint(-230, -120)
            ran_cen_3y = random.randint(120, 230)
            ran_cen_3 = [ran_cen_3x, ran_cen_3y]

            ran_cen_4x = random.randint(-230, -120)
            ran_cen_4y = random.randint(-230, -120)
            ran_cen_4 = [ran_cen_4x, ran_cen_4y]

            ran_cen_list = [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4]            
            eligible_list = []

            # To randomly initialize the car within a determined range
            for x in range(-300, 300):
                for y in range(-300, 300):

                    if x >= (ran_cen_1x - 110) and x <= (ran_cen_1x + 110) and y >= (ran_cen_1y - 110) and y <= (ran_cen_1y + 110):
                        continue

                    if x >= (ran_cen_2x - 110) and x <= (ran_cen_2x + 110) and y >= (ran_cen_2y - 110) and y <= (ran_cen_2y + 110):
                        continue

                    if x >= (ran_cen_3x - 110) and x <= (ran_cen_3x + 110) and y >= (ran_cen_3y - 110) and y <= (ran_cen_3y + 110):
                        continue

                    if x >= (ran_cen_4x - 110) and x <= (ran_cen_4x + 110) and y >= (ran_cen_4y - 110) and y <= (ran_cen_4y + 110):
                        continue

                    eligible_list.append((x,y))

            simulator = DrivingEnv('T2', eligible_list, render_mode=render_mode, config_filepath=config_filepath, ran_cen_list=ran_cen_list)
        
            # To keep track of the number of timesteps per episode
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset(eligible_list=eligible_list)
            ###########################################################

            # The following code is a basic example of the usage of the simulator
            road_status = False

            self.previous_state = state # S^t
            self.previous_action = np.array([1, 2]) # A^t
            self.mud_pits = ran_cen_list

            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):

                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                cur_time += 1

                state, reward, terminate, reached_road, info_dict = simulator._step(self.previous_action) # S^(t+1), R^t 
                fpsClock.tick(FPS)
                if terminate:
                    self.theta += (reward - np.dot(self.theta,task2_features(transition(self.previous_state, self.previous_action),self.mud_pits))/100)*task2_features(transition(self.previous_state, self.previous_action),self.mud_pits)/cur_time
                    road_status = reached_road
                    break
                action = self.next_action(state)
                self.theta += (reward + (np.dot(self.theta,task2_features(transition(state, action),self.mud_pits)) - np.dot(self.theta,task2_features(transition(self.previous_state, self.previous_action),self.mud_pits)))/100)*task2_features(transition(self.previous_state, self.previous_action),self.mud_pits)/cur_time
                self.previous_state = state
                self.previous_action = action
                theta_update(self.theta)

            print(str(road_status) + ' ' + str(cur_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    parser.add_argument("-t", "--task", help="task number", choices=['T1', 'T2'])
    parser.add_argument("-r", "--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("-m", "--render_mode", action='store_true')
    parser.add_argument("-f", "--frames_per_sec", help="fps", type=int, default=30) # Keep this as the default while running your simulation to visualize results
    args = parser.parse_args()

    config_filepath = args.config
    task = args.task
    random_seed = args.random_seed
    render_mode = args.render_mode
    fps = args.frames_per_sec

    FPS = fps

    random.seed(random_seed)
    np.random.seed(random_seed)

    if task == 'T1':
        
        agent = Task1()
        agent.controller_task1(config_filepath=config_filepath, render_mode=render_mode)

    else:

        agent = Task2()
        agent.controller_task2(config_filepath=config_filepath, render_mode=render_mode)
