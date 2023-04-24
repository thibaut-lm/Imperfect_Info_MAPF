import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the Network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1,padding=0),
            nn.ReLU(),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 512), #64 - 3136
            nn.ReLU(),
            nn.Linear(512, 5),
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x) # Flatten convolutional output
        #print(len(x))
        x = self.fc_layers(x)
        
        return x

def reinforce(env, num_episodes, gamma, policy_network=None):
    # Initialize the policy network
    if policy_network is None:
        policy_network = Network()
    
    # Set up the optimizer and the loss function
    optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create empty lists to store the episode agent rewards
    episodes_total_rewards_dict = {}
    
    # Loop over episodes
    for i_episode in tqdm(range(num_episodes)):
        # Initialize episode variables
        episode_rewards_dict = {}
        episode_log_probs_dict = {}
        episode_states_dict = {}
        
        # Reset the environment
        state_dict = env.reset()
        
        # Get the agents ids
        agent_ids = state_dict.keys()
        
        # Loop over timesteps
        done = False
        while not done:
            
            # Preprocess the states
            state_tensor_dict = {}
            for agent_id in agent_ids:
                state_tensor_dict[agent_id] = torch.from_numpy(state_dict[agent_id]).float().unsqueeze(0)
            
            # Get the logits from the policy network
            logits_dict = {}
            for agent_id in agent_ids:
                logits_dict[agent_id] = policy_network(state_tensor_dict[agent_id])
            
            # Sample actions and their probs from the logits
            action_dict = {}
            action_log_probs_dict = {}
            for agent_id in state_dict:
                action_probs = torch.softmax(logits_dict[agent_id], dim=0)
                action = np.random.choice(np.arange(5), p=action_probs.detach().numpy().squeeze())
                
                action_dict[agent_id] = {'movement': action, 'ask_communication':1}
                action_log_probs_dict[agent_id] = torch.log(action_probs[action])
            
            # Take the action in the environment
            next_state_dict, reward_dict, done_dict, _ = env.step(action_dict)
            # Append the reward, log probabilities, and states
            for agent_id in agent_ids:
                if agent_id in episode_rewards_dict : # Check if it's the not first turn
                    episode_rewards_dict[agent_id].append(reward_dict[agent_id])
                    episode_log_probs_dict[agent_id].append(action_log_probs_dict[agent_id])
                    episode_states_dict[agent_id].append(state_tensor_dict[agent_id])
                else:
                    episode_rewards_dict[agent_id]   = [ reward_dict[agent_id] ]
                    episode_log_probs_dict[agent_id] = [ action_log_probs_dict[agent_id] ]
                    episode_states_dict[agent_id]    = [ state_tensor_dict[agent_id] ]
            
            # Check if episode done
            done = False not in done_dict.values() or len(episode_rewards_dict[1]) > 1000
            
            if not done :
                # Update the state
                state_dict = next_state_dict
                
        ########################################################################## UPDATE NETWORK #########################################################################
        # Compute the discounted rewards
        discounted_rewards_dict = {}
        for agent_id in agent_ids:
            rewards = episode_rewards_dict[agent_id]
            discounted_rewards_dict[agent_id] = []
            R = 0
            for reward in reversed(rewards):
                R = reward + gamma * R
                discounted_rewards_dict[agent_id].insert(0, R)

        # Normalize the discounted rewards
        for agent_id in agent_ids:
            rewards = discounted_rewards_dict[agent_id]
            rewards = torch.tensor(rewards)
            rewards = (rewards - rewards.mean()) #/ (rewards.std() + 1e-9)
            discounted_rewards_dict[agent_id] = rewards
            
        optimizer.zero_grad()

        # Compute the loss for each agent in the episode
        agent_losses = []
        for agent_id in agent_ids:
            discounted_reward = discounted_rewards_dict[agent_id]
            log_probs = episode_log_probs_dict[agent_id]
            agent_loss = 0
            for log_prob, reward in zip(log_probs, rewards):
                agent_loss -= log_prob * reward
            agent_loss /= len(log_probs)
            agent_losses.append(agent_loss)
            
            if agent_id in episodes_total_rewards_dict:
                episodes_total_rewards_dict[agent_id].append(sum(episode_rewards_dict[agent_id]))
            else:
                episodes_total_rewards_dict[agent_id] = [ sum(episode_rewards_dict[agent_id]) ]
            
        # Compute the mean loss across all agents in the episode
        loss = torch.stack(agent_losses).mean()

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
    
    # Close the environment
    env.close()
    
    
    import statistics
    print(episodes_total_rewards_dict)
    num_episodes = len(episodes_total_rewards_dict[1])  # Assumes all agents have the same number of episodes

    mean_episode_rewards = []

    for i in range(num_episodes):
        episode_rewards = [episodes_total_rewards_dict[agent_id][i] for agent_id in episodes_total_rewards_dict]
        mean_episode_reward = statistics.mean(episode_rewards)
        mean_episode_rewards.append(mean_episode_reward)

    print(mean_episode_rewards)
        
    # Plot the cumulative rewards so far
    
    plt.plot(mean_episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode')
    plt.show()
    
    return policy_network
        