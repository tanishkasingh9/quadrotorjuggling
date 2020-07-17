import tensorflow as tf
import numpy as np
import vrep
import envi
import os.path
import main
from random import randint
my_path = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(my_path,'ppo/ProximalPolicy.ckpt')
SAVED_MODEL_PATH = os.path.join(my_path,'ppo/eval/ProximalPolicy.ckpt')
#LOG_FILE = os.path.join(my_path,'ppo/reward_episode.log')
class ProximalPolicy(object):

    def __init__(self,epochs, epoch_runs):
        self.prev_ball_pos = [0,0,0]
        self.batch_size = 1000
        self.epochs = epochs
        self.agentlr = 0.0001
        self.Teacherlr = 0.0002
        self.total_ep_rewards = []
        self.loss_list = []
        self.agentsteps= epoch_runs
        self.Teachersteps = epoch_runs
        self.l = 0.5
        self.kltarget =0.01

        



    def createmodel(self, label, ifTrainable):
        #Creates a tensorflow deep network
        #Is used here to create model for new policy and old policy, also mention if the created neural network is Trainable =True/false

        with tf.variable_scope(label):
            #creating f(n) and g(n) 
            inputLayer = tf.layers.dense(self.states, 100, tf.nn.relu, trainable=ifTrainable)
            cal_mu = 2 * tf.layers.dense(inputLayer, self.n_acts, tf.nn.tanh, trainable=ifTrainable) 
            cal_L = tf.layers.dense(inputLayer, self.n_acts, tf.nn.softplus, trainable=ifTrainable)
            line_distance_normal = tf.distributions.Normal(loc=cal_mu, scale=cal_L)
        parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=label)
        return line_distance_normal,parameters


    def checkKLclip(self,ol,ne,surr):

        #performing KL Divergence on the functions for the trust region
        self.L = tf.constant(self.l)
        

        #getting KL distributions for old and new functions 
        divfunc = tf.distributions.kl_divergence(ol,ne)

        #reduce mean on the calculated divfunc
        self.KLmean = tf.reduce_mean(divfunc)

        #calculate loss on the new and old policy parameters
        self.aloss = -(tf.reduce_mean(surr - self.L*divfunc))

        return self.aloss


    def stateact_update(self,state,action,reward):
        adv = self.session.run(self.adv, { self.states: state,self.cum_disc_r :reward})

        #update agent
        for i in range(self.agentsteps):
            self.session.run(self.old_updated_ac)
            i, divfunc = self.session.run([self.train_op,self.KLmean],{self.states:state, self.actions : action, self.advantage : adv, self.L : self.l})
            if divfunc > 4 * 0.01:
                break
        if divfunc<self.kltarget/1.5:
            self.l = self.l/2
        elif divfunc>self.kltarget*1.5:
            self.l*=2
        else: 
            [self.session.run(self.train_op,{self.states:state, self.actions : action, self.advantage : adv}) for i in range(self.agentsteps)]
        #four final clip
        self.l = np.clip(self.l,1e-4,10)


        #update the teacher
        [self.session.run(self.Teachertrain, {self.states: state, self.cum_disc_r: reward}) for i in range(self.Teachersteps)]



    def v_values_get(self, observation):
        if self.obs.ndim <2: observation = observation[np.newaxis,:]
        return self.session.run(self.v_values, {self.states : observation.reshape(1,-1)})[0,0]


    def  select_action(self, observation, n):
        observation = observation[np.newaxis,:]
        #print(observation.reshape(1,-1).shape())
        #print(type(self.sam))
        a = self.session.run(self.sam,{self.states:observation.reshape(1,-1)})[0]
        return a


    def train_one_epoch(self):
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_rew = []         # for measuring episode returns

        # reset episode-specific variables
        self.obs = self.env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep
        cum_rew_ep = 0
        # render first episode of each epoch
        # finished_rendering_this_epoch = False
        # collect experience by acting in the environment with current policy


        while True:
            # rendering
            #if (not finished_rendering_this_epoch) and render:
            #    env.render()

            # save obs
            # _, curr_ball_position = vrep.simxGetObjectPosition(env.clientId, env.ball, env.quad, vrep.simx_opmode_streaming)
            # if self.prev_ball_pos == curr_ball_position:
            #     continue
            # self.prev_ball_pos = curr_ball_position
            batch_obs.append(self.obs.copy())
            #print(obs)
            #get action

            self.action = self.select_action(self.obs,self.new)
            arr = np.where(self.action == np.amax(self.action))[0][0]
            #print(arr)
            #act in the environment
            done, obs_new, rew = self.env.step(arr)

            #save actions and rewards
            batch_acts.append(self.action)
            ep_rews.append(rew)

            #normalize with a const
            batch_rew.append((rew + 8)/8)

            #updating new state and calculating total reward
            self.obs = obs_new
            cum_rew_ep += rew



            if done:
                #the episode is over then record information about the episode
                #ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                r_discount = []

                #get V value for state
                v_val = self.v_values_get(self.obs)

                #rewards added to a list
                for rewards in batch_rew[::-1]:
                    v_ = rewards + 0.9*v_val
                    r_discount.append(v_val)
                r_discount.reverse()

                #reset environment
                self.obs, done, ep_rews = self.env.reset(), False, []

                #update the state, action and reward table
                s = np.vstack(batch_obs)
                a = np.vstack(batch_acts)
                r = np.array(r_discount)[:, np.newaxis]

               #print(s.shape,a.shape,r.shape)
                self.stateact_update(s,a,r)


            # end experience loop if we have enough of it
            if len(batch_obs) > self.batch_size:
                break
        return len(batch_obs),cum_rew_ep
 





    def train_get(self):
        hidden_sizes=[64,32]
        #starting environment
        self.env = envi.quadBounceSim()

        #hardcoding the continuous state space and discrete action space
        self.obs_dim = self.env.observation_dimensions
        self.n_acts = self.env.action_space

        #intializing session
        self.session = tf.Session()
        self.states = tf.placeholder(tf.float32, [None, self.obs_dim], 'state')

        #making a critic that calculates advantage
        with tf.variable_scope('Teacher'):
            inputLayer = tf.layers.dense(self.states, 100, tf.nn.relu)
            self.v_values = tf.layers.dense(inputLayer,1)

            #output defined from the v_values
            self.cum_disc_r = tf.placeholder(tf.float32, [None, 1], 'discountedreward')
            self.adv = self.cum_disc_r- self.v_values
            self.Tloss = tf.reduce_mean(tf.square(self.adv))

            #train the neural network Used later
            self.Teachertrain = tf.train.AdamOptimizer(self.Teacherlr).minimize(self.Tloss)

        #making an agent that learns
        self.new, self.newParameters = self.createmodel('new', True)
        self.old, self.oldParameters = self.createmodel('old', False)
        with tf.variable_scope('sam'):
            self.sam = tf.squeeze(self.new.sample(1), axis=0)       
        # choosing action
        with tf.variable_scope('old'):
            self.old_updated_ac = [o.assign(n) for n, o in zip(self.newParameters, self.oldParameters)]
        #defining action and action advantage
        self.actions = tf.placeholder(tf.float32, [None, self.n_acts], 'action')
        self.advantage= tf.placeholder(tf.float32, [None, 1], 'advantage')

        with tf.variable_scope('FinalLoss'):
            with tf.variable_scope('surrogate'):
                rat = self.new.prob(self.actions)/self.old.prob(self.actions)
                self.surrg = rat * self.advantage
            self.loss = self.checkKLclip(self.old,self.new,self.surrg)

        self.train_op = tf.train.AdamOptimizer(self.agentlr).minimize(self.loss)
        self.session.run(tf.global_variables_initializer())
        #self.session.run(tf.global_variables_initializer())

    def eval(self):
        self.train_get()
        env = envi.quadBounceSim()
        saver  = tf.train.Saver()
        saver.restore(self.session, SAVED_MODEL_PATH)
        no_of_runs = self.epochs
        


        for i in range (self.epochs):
            done = False
            while True:
                self.train_one_epoch()


    def train(self):
        self.train_get()
        for i in range(self.epochs):

            #for first episode
            batchlen, cum_rew_ep = self.train_one_epoch()
            if i ==0: self.total_ep_rewards.append(cum_rew_ep)
            else: self.total_ep_rewards.append(self.total_ep_rewards[-1]*0.9 + cum_rew_ep * 0.1)
            #print all outputs
            print 'epoch:',i
            self.saver  = tf.train.Saver()
            save_path = self.saver.save(self.session,MODEL_PATH)



 

    









