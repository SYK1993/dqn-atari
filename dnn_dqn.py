'''
The input is ram pixel value
'''
import numpy as np
import tensorflow as tf
import gym
from collections import deque
import random

env = 'FishingDerby-ram-v0'

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            replace_target_iter=1000,
            batch_size=32,
            output_graph=False,
            load_model=False, 
            train = True

    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size

        self.load_model = load_model
        self.train = train
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps

        self.lr = learning_rate
        # total learning step
        self.learn_step_counter = 0
        self.memory_size = 40000
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]

        self._build_net() # Declares for subsequent parameter initializations

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())
        # self.cost_his = []
        if self.load_model:
            self.restore_model()

    def _build_net(self):

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables

            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            if self.train:
                w_initializer, b_initializer = \
                    tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            else:
                w_initializer, b_initializer = None,None # No initialization is performed under test

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, 128],initializer=w_initializer,collections=c_names)
                b1 = tf.get_variable('b1', [1, 128], initializer=w_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [128, 128],initializer=w_initializer,collections=c_names)
                b2 = tf.get_variable('b2', [1, 128], initializer=w_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b1)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [128, 20], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, 20],initializer=w_initializer,collections=c_names)
                l3 = tf.nn.relu(tf.matmul(l2,w3)+b3)

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [20, self.n_actions], initializer=w_initializer,collections=c_names)
                b4 = tf.get_variable('b4', [1, self.n_actions], initializer=w_initializer,collections=c_names)
                self.q_eval = tf.matmul(l3, w4) + b4


        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, 128], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, 128], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [128, 128],initializer=w_initializer,collections=c_names)
                b2 = tf.get_variable('b2', [1, 128], initializer=w_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [128, 20], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, 20], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.relu(tf.matmul(l2,w3)+b3)

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [20, self.n_actions], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l3, w4) + b4

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        if self.train:
            if np.random.uniform() > self.epsilon:
                # forward feed the observation and get q value for every actions
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)
        else:
            actions_value = self.sess.run(self.q_eval,feed_dict={self.s: observation})

            action = np.argmax(actions_value)
        return action

    # def choose_action_test(self,observation):
    #     observation = observation[np.newaxis, :]
    #     l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
    #     l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
    #     q_eval = tf.matmul(l2, w3) + b3
    #
    #     if np.random.uniform() > self.epsilon:
    #         # forward feed the observation and get q value for every actions
    #         actions_value = self.sess.run(q_eval, feed_dict={self.s: observation})
    #         action = np.argmax(actions_value)
    #     else:
    #         action = np.random.randint(0, self.n_actions)
    #     return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })
        # q_eval_ = self.sess.run(self.q_eval,feed_dict={self.s: batch_memory[:, -self.n_features:]}) #ddqn
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)#compute q_target
        # q_target[batch_index, eval_act_index] = reward + self.gamma * q_next[np.argmax(q_eval_,axis=1)] # ddqn
        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        # self.cost_his.append(self.cost)

        # decreasing epsilon, the minimum value is 0.05
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
        self.learn_step_counter += 1

    def save_model(self,step):
        self.saver.save(self.sess,'./model-FishingDerby-ram-v0-dqn/',global_step=step)

    def restore_model(self):
        checkpoint = tf.train.get_checkpoint_state('./model-FishingDerby-ram-v0-dqn/')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Error...')

    def plot_reard(self,reward,):
        import matplotlib.pyplot as plt
        plt.plot( [(i+1)*100 for i in range(50)],reward)
        plt.ylabel('episode')
        plt.xlabel('reward')
        plt.show()

if __name__ == '__main__':
    env = gym.make(env)
    env = env.unwrapped
    RENDER = True
    train = True
    load_model = False #Whether to load the previous model
    print(env.action_space)
    print(env.observation_space)

    RL = DeepQNetwork(n_actions=env.action_space.n,
                      n_features=env.observation_space.shape[0],
                      learning_rate=0.0025,
                      replace_target_iter=1000,
                      load_model=load_model,train=train)

    total_steps = 0
    total_reward = 0
    total_reward_list = []
    if train:
        print('It is training mode right now')
        for i_episode in range(999999):
            observation = env.reset()
            ep_r = 0
            if i_episode!=0 and (i_episode)%500==0:
                RL.save_model(i_episode)
                print('For 100 episode ,the total reward is', total_reward/100)
                total_reward_list.append(total_reward/500)
                total_reward = 0

            for _ in range(random.randint(1, 50)):
                observation, _, _, _ = env.step(1)
            while True:
                if RENDER:
                    env.render()

                action = RL.choose_action(observation)
                # print(action)
                observation_, reward, done, info = env.step(action)

                RL.store_transition(observation, action, reward, observation_)

                ep_r += reward

                if total_steps > 50000:

                    RL.learn()

                if done:
                    print('episode: ', i_episode,
                          'ep_r: ', ep_r,
                          ' epsilon: ', round(RL.epsilon, 6)
                          )
                    total_reward += ep_r
                    if RL.epsilon > RL.epsilon_end:
                        RL.epsilon -= RL.epsilon_decay_step
                    break
                observation = observation_
                total_steps += 1
        # RL.plot_reard(total_reward_list)

    else:
        print('It is testing mode right now')
        for i_episode in range(100):
            observation = env.reset()
            ep_r = 0

            for _ in range(random.randint(1, 50)):
                observation, _, _, _ = env.step(1)

            while True:
                if RENDER:
                    env.render()
                print(observation.shape)
                action = RL.choose_action(observation)
                # print(action)

                observation_, reward, done, info = env.step(action)


                # print(action)
                ep_r += reward
                if done:
                    print('episode: ', i_episode,
                          'ep_r: ', ep_r,
                          )
                    total_reward += ep_r
                    break
                observation = observation_

        print('For 100 episode ,the average total reward is',total_reward/100)