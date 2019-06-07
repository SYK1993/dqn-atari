import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K
import keras.backend.tensorflow_backend as KTF

EPISODES = 50000


class DQNAgent:
    def __init__(self, action_size,train):
        self.render = True
        self.load_model = True
        self.train = train

        self.state_size = (84, 84, 4)
        self.action_size = action_size

        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        # parameters about training
        self.batch_size = 32
        self.train_start = 20000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.memory_size = 500000
        self.memory = deque(maxlen=self.memory_size)
        self.no_op_steps = 30

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth=False)
        # session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
        # # run_config = tf.estimator.RunConfig(**CONF.runconfig).replace(session_config=session_conf)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        config.gpu_options.allow_growth = False
        self.sess = tf.Session(config=config)
        KTF.set_session(self.sess)

        # self.sess = tf.InteractiveSession(config=session_conf)
        # K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save_model/dqn.h5")

    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if self.train:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            else:
                q_value = self.model.predict(history)
                return np.argmax(q_value[0])
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, history, action, reward, next_history, ):
        self.memory.append((history, action, reward, next_history, ))


    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self,done):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, = [], [],

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])


        target_value = self.target_model.predict(next_history)

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            if done:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    def save_model(self, name):
        self.model.save_weights(name)

    def restore_model(self, filename):
        self.model.load_weights(filename)

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


# 210*160*3(color) --> 84*84(mono)
# float --> integer (to reduce the size of replay memory)
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    # In case of BreakoutDeterministic-v3, always skip 4 frames
    # Deterministic-v4 version use 4 actions
    train = False #Training mode or test mode
    env = gym.make('FishingDerby-v0')
    agent = DQNAgent(action_size=env.action_space.n,train=train)

    scores, episodes, global_step = [], [], 0
    if train:
        for e in range(EPISODES):
            done = False
            # 1 episode = 5 lives
            step, score, = 0, 0,
            observe = env.reset()

            # just do nothing at the start of episode to avoid sub-optimal
            for _ in range(random.randint(1, agent.no_op_steps)):
                observe, _, _, _ = env.step(env.action_space.sample())

            # At start of episode, there is no preceding frame
            # So just copy initial states to make history
            state = pre_processing(observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                if agent.render:
                    env.render()
                global_step += 1
                step += 1

                # get action for the current history and go one step in environment
                action = agent.get_action(history)
                # change action to real_action

                observe, reward, done, _ = env.step(action)
                # pre-process the observation --> history
                next_state = pre_processing(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)

                agent.avg_q_max += np.amax(
                    agent.model.predict(np.float32(history / 255.))[0])

                # save the sample <s, a, r, s'> to the replay memory
                agent.replay_memory(history, action, reward, next_history,)
                # every some time interval, train model
                agent.train_replay(done)
                # update the target model with model
                if global_step % agent.update_target_rate == 0:
                    agent.update_target_model()

                score += reward

                history = next_history

                # if done, plot the score over episodes
                if done:
                    if global_step > agent.train_start:
                        stats = [score, agent.avg_q_max / float(step), step,
                                 agent.avg_loss / float(step)]
                        for i in range(len(stats)):
                            agent.sess.run(agent.update_ops[i], feed_dict={
                                agent.summary_placeholders[i]: float(stats[i])
                            })
                        summary_str = agent.sess.run(agent.summary_op)
                        agent.summary_writer.add_summary(summary_str, e + 1)

                    print("episode:", e, "  score:", score, "  memory length:",
                          len(agent.memory), "  epsilon:", round(agent.epsilon,3),
                          "  global_step:", global_step, "  average_q:",
                          round(agent.avg_q_max / float(step),6), "  average loss:",
                          round(agent.avg_loss / float(step),6))

                    agent.avg_q_max, agent.avg_loss = 0, 0

            if e % 100== 0 :
                agent.model.save_weights('./save_model/dqn_'+str(e)+'_.h5')
    else:
        agent.restore_model('./save_model/model_dqn.h5')
        score_total = 0
        for e in range(100):
            done = False
            score =  0
            observe = env.reset()
            for _ in range(random.randint(1, agent.no_op_steps)):
                observe, _, _, _ = env.step(1)
            state = pre_processing(observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))
            while not done:
                if agent.render:
                    env.render()
                action = agent.get_action(history)
                # print(action)
                observe, reward, done, info = env.step(action)
                next_state = pre_processing(observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state, history[:, :, :, :3], axis=3)
                score += reward
                history = next_history
                if done:
                    score_total += score+99
                    print("episode:", e, "  score:", score+99)
        print('之前模型最高得分是15.58,第400回合的模型')
        print('过去100回合的平均得分',score_total/100)