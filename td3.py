import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from game import CarEnv


# DDPG needs 2 models: Actor and Critic (2 of them)
# Critic learns to estimate values.
# Actor learns to choose actions that maximize those values.

# TD3 uses 2 critics. Why 2 critics?
# Two critics independently estimate Q(s,a). When their predictions differ, taking the minimum for the target reduces the chance of following an over‑optimistic value. This dramatically stabilises training.

# 6 total networks: actor, actor_target, critic_1, critic_1_target, critic_2, critic_2_target

# Actor, input = state, output = action. Action will be 1 float in the [-1, 1] range
class Actor(nn.Module):
    def __init__(self, in_nodes, h1_nodes, h2_nodes, out_nodes):
        super().__init__()

        self.fc1 = nn.Linear(in_nodes, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.out = nn.Linear(h2_nodes, out_nodes)

        # stored activations
        self.activations = {}

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        self.activations["input"] = state.detach().cpu() * 2
        x = F.relu(self.fc1(state))
        self.activations["h1"] = x.detach().cpu()
        x = F.relu(self.fc2(x))
        self.activations["h2"] = x.detach().cpu()
        x = F.tanh(self.out(x)) # tanh naturally outputs [-1, 1], matching our input steering velocity
        self.activations["out"] = x / 2 + 0.5
        return x


# Critic, input = (state + action), output = Q(s,a). Output will be 1 float, representing the Q-value
class Critic(nn.Module):
    def __init__(self, in_nodes, h1_nodes, h2_nodes, out_nodes):
        super().__init__()

        self.fc1 = nn.Linear(in_nodes, h1_nodes) # will have 1 more input neuron compared to the Actor
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.out = nn.Linear(h2_nodes, out_nodes)

        # stored activations
        self.activations = {}

    def forward(self, state, action):
        # Safe way: preserves grad history if already a tensor, converts if numpy
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        
        if action.dim() == 1:
            action = action.unsqueeze(1)  # [batch_size] -> [batch_size, 1]

        x = torch.cat([state.float(), action.float()], dim=1)  # shape: [batch_size, state_dim + action_dim]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


# Experience Replay
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        # transition (state, action, new_state, reward, terminated)
        self.memory.append(transition)
    
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


HIDDEN_NODES = 128

class CarTD3:
    # Hyperparameters (as class attributes)
    lr = 1e-3  # learning rate
    gamma = 0.99  # discount factor

    replay_memory_size = 70000  # size of replay memory
    mini_batch_size = 64  # size of the training data set sampled from replay memory

    # Neural network
    actor_optimizer = None  # initialized later as an instance attribute
    critic_optimizer = None  # initialized later as an instance attribute

    def train(self, episodes: int):
        env = CarEnv()
        env.FPS = 60
        num_states_for_actor = env.observation_space.for_actor
        num_states_for_critic = env.observation_space.for_critic
        env.init_neural_net([num_states_for_actor, HIDDEN_NODES, HIDDEN_NODES, 1])

        memory = ReplayMemory(self.replay_memory_size)

        actor = Actor(num_states_for_actor, HIDDEN_NODES, HIDDEN_NODES, 1)
        actor_target = Actor(num_states_for_actor, HIDDEN_NODES, HIDDEN_NODES, 1)
        critic_1 = Critic(num_states_for_critic, HIDDEN_NODES, HIDDEN_NODES, 1)
        critic_1_target = Critic(num_states_for_critic, HIDDEN_NODES, HIDDEN_NODES, 1)
        critic_2 = Critic(num_states_for_critic, HIDDEN_NODES, HIDDEN_NODES, 1)
        critic_2_target = Critic(num_states_for_critic, HIDDEN_NODES, HIDDEN_NODES, 1)

        # Make the target and policy networks the same (copy weights/biases from one network to another)
        actor_target.load_state_dict(actor.state_dict())
        critic_1_target.load_state_dict(critic_1.state_dict())
        critic_2_target.load_state_dict(critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        self.critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=1e-2)
        self.critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=1e-2)

        # Keep track of total reward evolution
        total_reward_per_episode = np.zeros(episodes)

        # Track number of steps taken. Used for syncing policy => target network.
        sync_step_count = 0
        total_step_count = 0
        noise_std = 0.2

        self.actor_critic_update_delay = 1  # update actor & targets every 2 critic updates
        self.actor_critic_update_counter = 0

        consecutive_successful_episodes = 0

        # Episodes loop
        for episode in range(episodes):
            env.FPS = env.FPS * 1.05
            print(f"Episode: {episode}")

            render_this_episode = 0 <= (episode + 1) % 50 < 3
            # render_this_episode = True
            if render_this_episode:
                env.init_display()
            else:
                env.close_display()

            state = env.reset()
            done = False
            truncated = False

            steps_at_episode_start = total_step_count

            while (not done and not truncated):
                if render_this_episode:
                    env.episode = episode
                    env.render()

                # Choose action
                with torch.no_grad():
                    action = actor(state)[0].item()

                # add noise
                # mean = 0 (centered at zero),
                # std = 0.1 (~68% of sampled values fall between -0.1 and 0.1, ~95% between -0.2 and 0.2)
                noise = np.random.normal(0, noise_std)
                
                action += noise
                action = np.clip(action, -1, 1) # restrict between [-1, 1]
                
                # Apply action
                next_state, reward, done, truncated, _ = env.step(action, truncate_at=1500)

                # Increase the total reward for this episode
                total_reward_per_episode[episode] += reward

                # Save memory for Experience Replay
                memory.append((state, action, next_state, reward, done))

                # Train every step
                # if total_step_count % 4 == 0 and len(memory) > self.mini_batch_size:
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(
                        mini_batch,
                        actor,
                        actor_target,
                        critic_1,
                        critic_1_target,
                        critic_2,
                        critic_2_target
                    )

                env.animate_neural_net(actor.activations)
                
                # Move to next state
                state = next_state

                total_step_count += 1
                sync_step_count += 1
                noise_std = max(0.01, noise_std * 0.99999)
                
                # sync online and target networks using Polyak averaging
                tau = 0.5

                if sync_step_count > 50:
                    # sync Actor network
                    for target_param, param in zip(
                        actor_target.parameters(),
                        actor.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data
                            + (1 - tau) * target_param.data
                        )
                    
                    # sync Critic networks
                    for target_param, param in zip(
                        critic_1_target.parameters(),
                        critic_1.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data
                            + (1 - tau) * target_param.data
                        )
                    
                    for target_param, param in zip(
                        critic_2_target.parameters(),
                        critic_2.parameters()
                    ):
                        target_param.data.copy_(
                            tau * param.data
                            + (1 - tau) * target_param.data
                        )
                    
                    sync_step_count = 0

            # --- episode done ---
            steps_this_episode = total_step_count - steps_at_episode_start
            print(f"Steps this episode: {steps_this_episode}, noise std = {noise_std}")

            if steps_this_episode >= 1500:
                consecutive_successful_episodes += 1
            else: 
                consecutive_successful_episodes = 0
            
            if consecutive_successful_episodes >= 7:
                print("Consecutive succesful episodes")
                break

            # Decrease noise
            # noise_std = max(
            #     0.05,
            #     noise_std * 0.9995
            # )

        # --- all episodes done ---
        print(f"Training done. Ran for {episodes} episodes")
        print(f"Total training steps: {total_step_count}")
        torch.save(actor.state_dict(), "car_td3_actor_test.pt")
        torch.save(critic_1.state_dict(), "car_td3_critic_1_test.pt")
        torch.save(critic_2.state_dict(), "car_td3_critic_2_test.pt")

        # Create a graph
        plt.figure(1)
        plt.plot(total_reward_per_episode)
        plt.savefig('car_ddpg_total_reward.png')


    # optimize policy network
    def optimize(self, mini_batch, actor, actor_target, critic_1, critic_1_target, critic_2, critic_2_target):
        states, actions, next_states, rewards, dones = zip(*mini_batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # .unsqueeze(1) — reshapes both from [64] (one dim array) -> [64, 1] (two dim array with 64 rows with 1 cols)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # --- Critic update part ---
        with torch.no_grad():
            # compute target actions (actions to be done in the next state)
            next_actions = actor_target(next_states)

            # Target Policy Smoothing (adding noise)
            # Returns a tensor with the same size as input that is filled with
            # random numbers from a normal distribution with mean 0 and standard deviation 1
            noise = torch.randn_like(next_actions) * 0.01  # Gaussian noise, std 0.2
            # Clamp -> elements below the minimum are set to the minimum, and those above the maximum are set to the maximum
            noise = noise.clamp(-0.02, 0.02)  # clip the noise
            # smoothed_next_actions = (next_actions + noise).clamp(-1, 1)  # stay within action bounds
            smoothed_next_actions = next_actions # no noise added for better learning in this case

            # target Q values (how good are those next actions), using the Target networks as a stable anchor point
            # we evaluate the target Q values for both critic networks, and use the minimum value
            next_q1 = critic_1_target(next_states, smoothed_next_actions)
            next_q2 = critic_2_target(next_states, smoothed_next_actions)
            target_q = torch.min(next_q1, next_q2) # clipped double-Q
            # Taking the minimum gives a more conservative estimate. If one critic overestimates, the other likely does not

            # Bellman target. If done, target_q will be just = rewards
            target_q = rewards + self.gamma * (1 - dones) * target_q  # [64, 1]

        # current Q values, using the online critics
        current_q1 = critic_1(states, actions)
        current_q2 = critic_2(states, actions)

        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        self.critic_2_optimizer.step()

        self.actor_critic_update_counter += 1

        # At this point the critics have learned a slightly better approximation of Q(s,a)

        # Why delayed updates? The actor relies on an accurate Q‑function. If the critic is updated every step but the actor is updated equally often, the actor can over‑react to a momentarily poor critic. Updating the actor less often gives the critics time to “settle” before the policy uses them.
        if self.actor_critic_update_counter % self.actor_critic_update_delay == 0:
            # --- Actor update part ---
            # The objective is: for each state, choose an action that maximises the Q‑value estimated by the critic. We therefore want to perform gradient ascent on Q(s, (actor(s)))

            # Let actor choose actions
            pred_actions = actor(states) # These are the actions the actor currently thinks are best

            # Ask critic how good they are
            q_values = critic_1(states, pred_actions) # could use either of critics
            actor_loss = -q_values.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # # Soft-update (Polyak averaging) all four target networks with small tau
            # tau = 0.005
            # for target_param, param in zip(actor_target.parameters(), actor.parameters()):
            #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            # for target_param, param in zip(critic_1_target.parameters(), critic_1.parameters()):
            #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            # for target_param, param in zip(critic_2_target.parameters(), critic_2.parameters()):
            #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def test(self):
        env = CarEnv()
        num_states_for_actor = env.observation_space.for_actor

        env.init_neural_net([num_states_for_actor, HIDDEN_NODES, HIDDEN_NODES, 1])
        env.FPS = 120

        actor = Actor(num_states_for_actor, HIDDEN_NODES, HIDDEN_NODES, 1)
        # critic = Critic(num_states_for_critic, HIDDEN_NODES, HIDDEN_NODES, 1)

        actor.load_state_dict(torch.load("car_td3_actor_test.pt"))

        while True:
            env.init_display()
            state = env.reset()
            done = False
            truncated = False

            while (not done and not truncated):
                env.episode = None
                env.render()
                # Exploit with the best action in current state

                with torch.no_grad():
                    action = actor(state)[0].item()
                
                env.animate_neural_net(actor.activations)
                
                # Apply action
                next_state, reward, done, truncated, _ = env.step(action, truncate_at=10000)
                
                # Move to next state
                state = next_state


car_td3 = CarTD3()
# car_td3.train(episodes=200)
car_td3.test()