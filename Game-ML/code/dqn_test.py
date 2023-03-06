from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym

seed = 42

model = keras.models.load_model('/content/drive/MyDrive/game_ai/train_model', compile=False)

env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

env = gym.wrappers.Monitor(env, '/content/drive/MyDrive/game_ai/videos', video_callable=lambda episode_id: True,
    force=True)

n_episodes = 10
returns = np.zeros(n_episodes, dtype=float)

for i in range(n_episodes):
  ret = 0

  state = np.array(env.reset())
  done = False

  while not done:

    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    action_values = model.predict(state_tensor)
    action = np.argmax(action_values)
    state_next, result, done, _ = env.step(action)
    state = np.array(state_next)
    returns[i] += result

env.close()

print('Returns: {}'.format(returns))