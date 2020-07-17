
import envi
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow.keras import losses as loss_fn
import tensorflow.keras.backend as keras_backend

class VanillaPolicyGradient(object):
    def __init__(self, env):
        self.input_shape = [env.observation_dimensions]
        self.input_size = self.input_shape[0]
        self.output_dim = env.action_space
        self.hidden_dims = [64, 32]
        self.losses = []
        self.GAMMA = 0.9
        self.LEARNING_RATE = 0.01

        self.build_neural_net()
  
        
    def build_neural_net(self):
        input_layer = tf.keras.layers.Input(shape = self.input_shape,name="input")
        advantage = tf.keras.layers.Input(shape=[1],name="advantage")

        model = input_layer
        for dim in self.hidden_dims:
            # model = tf.keras.layers.Dense(dim,activation='tanh')(model)
            model = tf.keras.layers.Dense(dim,activation='relu')(model)
        output_layer = tf.keras.layers.Dense(self.output_dim,activation='softmax')(model)

        # def vpg_loss(y_true, y_pred):
        #     likelihood = y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred)
        #     log_likelihood = keras_backend.log(likelihood)
        #     loss = keras_backend.mean(log_likelihood * advantage, keepdims=True)
        #     return loss

        self.model_train = tf.keras.Model(inputs=[input_layer,advantage],outputs=output_layer)
        self.model_train.compile(loss=loss_fn.msle,optimizer=tf.keras.optimizers.Adam(lr=self.LEARNING_RATE))
        # self.model_train.compile(loss=loss_fn.kullback_leibler_divergence,optimizer=tf.keras.optimizers.Adam(lr=self.LEARNING_RATE))
        # self.model_train.compile(loss=loss_fn.categorical_crossentropy,optimizer=tf.keras.optimizers.Adam(lr=self.LEARNING_RATE))

        self.model_predict = tf.keras.Model(inputs=[input_layer],outputs=output_layer)
    

    def get_discounted_rewards(self,reward_lst):
        prev_val = 0
        out = []
        for val in reward_lst:
            # print(val)
            new_val = val + prev_val * self.GAMMA
            # print(new_val)
            out.append(new_val)
            prev_val = new_val

        # return np.array(out)
        # print(np.array(out))
        return np.array(out[::-1])
    

    def fit(self,state_buffer,action_buffer,reward_buffer):
        # print(action_buffer)
        # print(state_buffer)
        discounted_rewards = reward_buffer
        # discounted_rewards -= keras_backend.mean(discounted_rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        # print(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        # print(discounted_rewards)

        actions_train = np.zeros([len(action_buffer), self.output_dim])
        actions_train[np.arange(len(action_buffer)), action_buffer] = 1

        loss = self.model_train.train_on_batch([state_buffer, discounted_rewards], actions_train)
        # print(loss)
        self.losses.append(loss)
        return loss
    

    def get_action(self,state):
        action_prob = np.squeeze(self.model_predict.predict(state))
        # print(self.model_predict.predict(state))
        # print(action_prob)
        return np.random.choice(range(self.output_dim),p=action_prob)

    def save_network_weights(self, save_location):
        self.model_train.save_weights(save_location+'/training_model_weights.h5')
        self.model_predict.save_weights(save_location+'/predicting_model_weights.h5')

    def load_network_weights(self, load_location):
        self.model_train.load_weights(load_location+'/training_model_weights.h5')
        self.model_predict.load_weights(load_location+'/predicting_model_weights.h5')


class QuadBounceBallVPG(object):

    def __init__(self, total_episodes=1000, steps_per_episode=200):
        self.total_episodes = int(total_episodes)
        self.steps_per_episode = int(steps_per_episode)
        # print("init", total_episodes, steps_per_episode)

    # initialize env and call step////
    def execute_episode(self,env,agent,n):
        #reset buffer every episode
        state_buffer,action_buffer,reward_buffer = [],[],[] # bugger to strore values every step
        total_reward = 0

        for episode in range(n):
            done = False        
            current_state = env.reset()
            # step 
            while not done:
                action_to_execute = agent.get_action(np.reshape(current_state,[1,agent.input_size])) 
                done,next_state,reward = env.step(action_to_execute)

                total_reward += reward
                state_buffer.append(current_state)
                action_buffer.append(action_to_execute)
                reward_buffer.append(reward)
                current_state = next_state
        # store episode in buffer
        state_buffer = np.array(state_buffer)
        action_buffer = np.array(action_buffer)
        reward_buffer = np.array(agent.get_discounted_rewards(reward_buffer)) 

        loss = agent.fit(state_buffer,action_buffer,reward_buffer)
        print("Training on batch complete. Loss: "+ str(loss))
        return total_reward/n



    def train(self):
        log_file = open('vpg_reward_episode.log','w')

        env = envi.quadBounceSim()
        agent = VanillaPolicyGradient(env)

        for episode in range(self.total_episodes):
            avg_reward = self.execute_episode(env, agent, self.steps_per_episode)
            # print("avg_reward:", avg_reward)
            agent.save_network_weights("vpg")
            print("episode: "+ str(episode)+", reward: "+str(avg_reward))
            log_file.write("episode: "+ str(episode)+", reward: "+str(avg_reward)+"\n")
        env.destroy()


    def eval(self):
        env = envi.quadBounceSim()
        agent = VanillaPolicyGradient(env)

        agent.load_network_weights("vpg/eval")
        for i in range(self.total_episodes):
            done = False
            total_reward = 0
            current_state = env.reset()
            while not done:
                action_to_execute = agent.get_action(np.reshape(current_state,[1,agent.input_size]))
                # print(action_to_execute)
                done,next_state,reward = env.step(action_to_execute)
                total_reward += reward
                current_state = next_state
            print("reward obtained  "+str(total_reward))

# QuadBounceBallVPG().eval()