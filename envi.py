import vrep
import sys
import numpy as np
import time
import random

class quadBounceSim(object):

    def __init__(self):
        vrep.simxFinish(-1)
        self.clientId = vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
        if self.clientId != -1:
            print('Connected to remote API server')
        else:
            print('Connection to V-Rep server failed')
            sys.exit('Could Not Connect')

        vrep.simxSynchronous(self.clientId,True)
        self.bounceCount = 0
        vrep.simxStopSimulation(self.clientId, vrep.simx_opmode_oneshot)

        self.get_handles()
        self.reset()

        k = 0.02
        self.move_dict = {0:[k,0,0],1:[-k,0,0],2:[0,k,0],3:[0,-k,0],4:[0,0,2*k],5:[0,0,-2*k]}
        self.observation_dimensions = 7
        self.action_space = 6
        self.incremental_step_count  = 0
        self.incremental_step = []
        self.prev_ball_pos = []
        self.bounceFlag = False
        self.prevBounceFlag = False
        

    def get_handles(self):
        _, self.quad = vrep.simxGetObjectHandle(self.clientId, "Quadricopter", vrep.simx_opmode_blocking)
        _, self.quad_target = vrep.simxGetObjectHandle(self.clientId, "Quadricopter_target", vrep.simx_opmode_blocking)
        _, self.ball = vrep.simxGetObjectHandle(self.clientId, "ball", vrep.simx_opmode_blocking)
        _, self.quad_position = vrep.simxGetObjectPosition(self.clientId, self.quad, -1, vrep.simx_opmode_streaming)
        _, self.quad_orientation = vrep.simxGetObjectOrientation(self.clientId, self.quad, -1, vrep.simx_opmode_streaming)
        _, self.quad_target_position = vrep.simxGetObjectPosition(self.clientId, self.quad_target, -1, vrep.simx_opmode_streaming)
        _, self.ball_position = vrep.simxGetObjectPosition(self.clientId, self.ball, self.quad, vrep.simx_opmode_streaming)
        #_, self.collision_stream = vrep.simxReadCollision(self.clientId,self.quad_collision_handle,vrep.simx_opmode_streaming)

    def destroy(self):
        vrep.simxStopSimulation(self.clientId, vrep.simx_opmode_oneshot)

    def reset(self):
        vrep.simxStopSimulation(self.clientId, vrep.simx_opmode_oneshot)
        quad_pos = [-0.75, 0.1, 0.5]
        random.seed(time.time())
        ball_pos = [-0.75+random.uniform(-0.05,0.05), 0.1+random.uniform(-0.05,0.05), 1.6]
        quad_target_pos = [-0.75, 0.1, 0.5]
        vrep.simxSetObjectPosition(self.clientId, self.quad_target, -1, quad_target_pos, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectPosition(self.clientId, self.quad, -1, quad_pos, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectPosition(self.clientId, self.ball, -1, ball_pos, vrep.simx_opmode_oneshot)
        
        time.sleep(0.6)
        vrep.simxStartSimulation(self.clientId, vrep.simx_opmode_oneshot)
        
        _, self.ball_position = vrep.simxGetObjectPosition(self.clientId, self.ball, self.quad, vrep.simx_opmode_buffer)
        _, self.quad_position = vrep.simxGetObjectPosition(self.clientId, self.quad, -1, vrep.simx_opmode_buffer)
        _, self.quad_orientation = vrep.simxGetObjectOrientation(self.clientId, self.quad, -1, vrep.simx_opmode_buffer)
        _, self.quad_target_position = vrep.simxGetObjectPosition(self.clientId, self.quad_target, -1, vrep.simx_opmode_buffer)
        self.previous_target_pos = np.asarray(self.quad_target_position)
        self.prev_ball_pos = self.ball_position

        # return np.concatenate((np.asarray(self.quad_position), np.asarray(self.ball_position)),axis=0)
        return np.asarray(self.ball_position+ [self.quad_position[2]]+ self.quad_orientation)

        # not used currently
    def random_init(self):
        x_range = [1.0, -1.0]
        y_range = [1.0, -1.0]
        z_range = [1.5, 0.4]
        quad_xyz = self.quad_position
        quad_target_xyz = self.quad_target_position
        ball_xyz = self.ball_position

        vrep.simxSetObjectPosition(self.clientId, self.quad, -1, quad_xyz, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectPosition(self.clientId, self.ball, -1, ball_xyz, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectPosition(self.clientId, self.quad_target, -1, quad_target_xyz, vrep.simx_opmode_oneshot)

    def step(self, action):
        #The Simulator returns the same state multiple times. Hence, wait until there is state change and then call exec_step
        while True:
            _, curr_ball_position = vrep.simxGetObjectPosition(self.clientId, self.ball, self.quad, vrep.simx_opmode_streaming)
            if self.prev_ball_pos == curr_ball_position:
                continue
            break
        
        d,obs,reward =  self.exec_step(action)
        self.prev_ball_pos = curr_ball_position
        return d,obs,reward


    def exec_step(self,action):
        # new target is equal to previous target + step taken
        new_target = list(self.previous_target_pos + np.asarray(self.move_dict[action]))
        vrep.simxSetObjectPosition(self.clientId, self.quad_target, -1, new_target, vrep.simx_opmode_oneshot)


        _, self.quad_position = vrep.simxGetObjectPosition(self.clientId, self.quad, -1, vrep.simx_opmode_buffer)
        _, self.quad_orientation = vrep.simxGetObjectOrientation(self.clientId, self.quad, -1, vrep.simx_opmode_buffer)
        _, self.quad_target_position = vrep.simxGetObjectPosition(self.clientId, self.quad_target, -1, vrep.simx_opmode_buffer)
        _, self.ball_position = vrep.simxGetObjectPosition(self.clientId, self.ball, self.quad, vrep.simx_opmode_buffer)

        self.previous_target_pos = np.asarray(new_target)
        observations = np.asarray(self.ball_position+ [self.quad_position[2]]+ self.quad_orientation)
        done, reward = self.get_reward()

        return done, observations, reward

    def get_reward(self):
        done = False
        reward = 0
        
        #Check if ball is going up or down
        if self.prev_ball_pos[2] + 0.005 < self.ball_position[2]:
            self.bounceFlag = True
        else:
            self.bounceFlag = False
        self.prev_ball_pos = self.ball_position

        # Count Number of Bounces and Reward for each bounce
        if self.bounceFlag == True and self.prevBounceFlag == False:
            reward += 500
            self.bounceCount += 1
        self.prevBounceFlag = self.bounceFlag

        #Reward quadcopter for being as closer to x,y position of ball
        euc_dist_xy_sq = np.sum(np.square(np.asarray(self.ball_position)[:-1]))
        reward = reward + min(10,0.5/(euc_dist_xy_sq+0.01))

        
        euc_dist_z_sq = self.ball_position[2]**2

        if euc_dist_xy_sq < 0.02:
            if self.bounceFlag == False:
                #When ball is falling, reward is inversely prop to distance between quad and ball
                if self.previous_target_pos[2] <  self.quad_target_position[2]:
                    reward += min(50,1/(euc_dist_z_sq+0.001))
            else:
                if self.quad_position[2] > 0.5 and self.previous_target_pos[2] > self.quad_target_position[2] :
                    reward += min(50,0.1/((self.quad_position[2]-0.5)**2+0.001))

        #Terminate state if ball has fallen below the quad or if it has fallen too far
        if self.ball_position[2]<0 or euc_dist_xy_sq > 0.3:
            print('Number of bounces: ', self.bounceCount)
            self.bounceCount = 0
            reward = -1000
            done = True
        return done,reward

    # Take 1
    # def get_reward(self):
    #     done = False
    #     reward = 0
        
    #     #Check if ball is going up or down
    #     if self.prev_ball_pos[2] + 0.005 < self.ball_position[2]:
    #         self.bounceFlag = True
    #     else:
    #         self.bounceFlag = False
    #     self.prev_ball_pos = self.ball_position

    #     # Count Number of Bounces and Reward for each bounce
    #     if self.bounceFlag == True and self.prevBounceFlag == False:
    #         reward += 500*self.bounceCount
    #         self.bounceCount += 1
    #     self.prevBounceFlag = self.bounceFlag

    #     #Reward quadcopter for being as closer to x,y position of ball
    #     euc_dist_xy_sq = np.sum(np.square(np.asarray(self.ball_position)[:-1]))
    #     reward = reward + min(10,0.5/(euc_dist_xy_sq+0.01))

    #     #Terminate state if ball has fallen below the quad or if it has fallen too far
    #     if self.ball_position[2]<0 or euc_dist_xy_sq > 0.3:
    #         print('Number of bounces: ', self.bounceCount)
    #         self.bounceCount = 0
    #         reward = -1000
    #         done = True
    #     return done,reward

    # Take 2
    # def get_reward(self):
    #     done = False
    #     reward = 10 # reward for each time step alive
        
    #     #Check if ball is going up or down
    #     if self.prev_ball_pos[2] + 0.005 < self.ball_position[2]:
    #         self.bounceFlag = True
    #     else:
    #         self.bounceFlag = False
    #     self.prev_ball_pos = self.ball_position

    #     # Count Number of Bounces and Reward for each bounce
    #     if self.bounceFlag == True and self.prevBounceFlag == False:
    #         reward += 50*self.bounceCount
    #         self.bounceCount += 1
    #     self.prevBounceFlag = self.bounceFlag

    #     #Reward quadcopter for being as closer to x,y position of ball
    #     euc_dist_xy_sq = np.sum(np.square(np.asarray(self.ball_position)[:-1]))
    #     reward = reward + min(10,0.5/(euc_dist_xy_sq+0.01)) #
    #     # print("Move xy:",euc_dist_xy_sq,min(10,0.1/(euc_dist_xy_sq+0.01)))

    #     #Reward positively when ball is falling down and quad is going up
    #     euc_dist_z_sq = self.ball_position[2]**2
    #     # if euc_dist_xy_sq < 0.05:
    #     #     if self.bounceFlag == False:
    #     #         if self.previous_target_pos[2] < self.quad_target_position[2]:
    #     #             reward+= min(10,10/(euc_dist_z_sq+0.001))
    #     #             print('Reward: go up1',euc_dist_z_sq,min(10,10/(euc_dist_z_sq+0.001)))
    #     #         else:
    #     #             reward-= min(10,10/(euc_dist_z_sq+0.001))
    #     #             print('Reward: go up2',euc_dist_z_sq,-min(10,10/(euc_dist_z_sq+0.001)))
                
    #     #     else:
    #     #         if self.previous_target_pos[2] > self.quad_target_position[2]:
    #     #             reward+= min(10,100*abs(self.quad_position[2]-0.5))
    #     #             print('Go down1:',self.quad_position[2]-0.5 ,min(10,100*abs(self.quad_position[2]-0.5)))
    #     #         else:
    #     #             reward-= min(10,100*abs(self.quad_position[2]-0.5))
    #     #             print('Go down2:',self.quad_position[2]-0.5 ,-min(10,100*abs(self.quad_position[2]-0.5)))
        

    #     #Terminate state if ball has fallen below the quad or if it has fallen too far
    #     if self.ball_position[2]<0 or euc_dist_xy_sq > 0.3:
    #         print('Number of bounces: ', self.bounceCount)
    #         self.bounceCount = 0
    #         reward = -1000
    #         done = True
    #     # print('Reward = ',reward)
    #     return done,reward