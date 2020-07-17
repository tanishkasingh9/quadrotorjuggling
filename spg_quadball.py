import tensorflow as tf
import numpy as np
import envi
import os.path


my_path = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(my_path,'spg/simple_pg.ckpt')
SAVED_MODEL_PATH = os.path.join(my_path,'spg/eval/simple_pg.ckpt')
LOG_FILE = os.path.join(my_path,'spg/reward_episode.log') 


class SimplePolicyGradient:
    def __init__(self, epochs=5000, runs_per_epoch=50, hidden_layers=[64,32], lr=1e-3):
        self.env = envi.quadBounceSim()
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.epochs = epochs
        self.runs_per_epoch = runs_per_epoch
        self.init_neural_network()

    def init_neural_network(self):
        observation_dimensions = self.env.observation_dimensions
        number_of_actions = self.env.action_space

        # neural network that decides the policy
        self.observations_ph = tf.placeholder(shape=(None, observation_dimensions), dtype=tf.float32)
        logits = self.multi_layer_perceptron(self.observations_ph, sizes=self.hidden_layers+[number_of_actions])

        self.actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)

        # gradient of loss function is policy gradient
        self.weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
        self.action_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
        action_masks = tf.one_hot(self.action_ph, number_of_actions)
        log_probabilities = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
        self.loss = -tf.reduce_mean(self.weights_ph * log_probabilities)

        # AdamOptimizer used for training neural network
        self.adam_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def multi_layer_perceptron(self,x, sizes, activation=tf.nn.tanh, output_activation=tf.nn.tanh):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = tf.layers.dense(x, units=size, activation=activation)
        return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

    def train(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        log_file = open(LOG_FILE,'w')
        # training loop
        for i in range(self.epochs):

            batch_loss, return_buffer, episode_lengths_list = self.run_epoch()
            print('epoch: '+str(i)+'\t loss: '+str(batch_loss)+'\t return: '+str(np.mean(return_buffer))+'\t episode_length: '+str(np.mean(episode_lengths_list)))
            log_file.write('epoch: '+str(i)+'\t loss: '+str(batch_loss)+'\t return: '+str(np.mean(return_buffer))+'\t episode_length: '+str(np.mean(episode_lengths_list))+'\n')
            save_path = saver.save(self.sess, MODEL_PATH)
        log_file.close()
        self.env.destroy()

    def run_epoch(self):
        # make some empty lists for logging.
        env = self.env
        observation_buffer = []          # to store observations
        action_buffer = []               # to store actions executed
        weights_buffer = []              # store weight for each log probability
        return_buffer = []               # to store overall reward per epoch
        episode_lengths_list = []        # for storing episode lengths

        # Initial Reset
        obs = env.reset()       
        done = False                     # Variable that checks if episode is over
        episode_rewards = []             # to record list of rewards

        for i in range(self.runs_per_epoch):
            while not done:
                observation_buffer.append(obs.copy())
                act = self.sess.run(self.actions, {self.observations_ph: obs.reshape(1,-1)})[0]
                done, obs, rew = env.step(act)

                action_buffer.append(act)
                episode_rewards.append(rew)

            episode_returns, episode_length = sum(episode_rewards), len(episode_rewards)
            return_buffer.append(episode_returns)
            episode_lengths_list.append(episode_length)

            # the weight for each logprob(a|s)
            weights_buffer += [episode_returns] * episode_length

            obs, done, episode_rewards = env.reset(), False, []

        # single policy gradient update step
        batch_loss, _ = self.sess.run([self.loss, self.adam_opt],
                                 feed_dict={
                                    self.observations_ph: np.array(observation_buffer),
                                    self.action_ph: np.array(action_buffer),
                                    self.weights_ph: np.array(weights_buffer)
                                 })
        return batch_loss, return_buffer, episode_lengths_list

    def eval(self):
        env= self.env
        self.sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(self.sess, SAVED_MODEL_PATH)
        no_of_runs = self.epochs
        for i in range(no_of_runs):
            done = False
            obs = env.reset()
            while not done:
                act = self.sess.run(self.actions, {self.observations_ph: obs.reshape(1,-1)})[0]
                done, obs, rew = env.step(act)
        self.env.destroy()