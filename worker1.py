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
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x) # Flatten convolutional output
        x = self.fc_layers(x)
        
        return x

def reinforce(env, num_episodes, gamma, policy_network=None):
    # Initialize the policy network
    if policy_network is None:
        policy_network = Network()
    
    # Set up the optimizer and the loss function
    optimizer = optim.Adam(policy_network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create empty lists to store the episode and cumulative rewards
    episodes_total_rewards = []
    
    # Loop over episodes
    for i_episode in tqdm(range(num_episodes)):
        # Initialize episode variables
        episode_rewards = []
        episode_log_probs = []
        episode_states = []
        
        # Reset the environment
        state = env.reset()[1]
        
        # Loop over timesteps
        done = False
        while not done:
            # Preprocess the state
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            
            # Get the logits from the policy network
            logits = policy_network(state_tensor)
            
            # Sample an action from the logits
            action_probs = torch.softmax(logits, dim=0)
            action = np.random.choice(np.arange(5), p=action_probs.detach().numpy().squeeze())
            joint_action = {1: {'movement': action, 'ask_communication': 1}}
            
            # Take the action in the environment
            next_state, reward, dones, _ = env.step(joint_action)
            
            # Append the reward, log probabilities, and states
            episode_rewards.append(reward[1])
            episode_log_probs.append(torch.log(action_probs[action]))
            episode_states.append(state_tensor)
            
            # Check if episode done
            done = False not in dones.values()
            
            if not done:
                # Update the state
                state = next_state[1]
            
            # Too long episode, manually reset episode
            if(len(episode_rewards) > 40):
                state = env.reset()[1]
                done = False
                episode_rewards = []
                episode_log_probs = []
                episode_states = []
                
            
        # Compute the discounted rewards
        discounted_rewards = []
        R = 0
        for reward in reversed(episode_rewards):
            R = reward + gamma * R
            discounted_rewards.insert(0,R)
            
        # Normalize the discounted rewards
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean())
        
        # Compute the loss
        loss = 0
        for log_prob, reward in zip(episode_log_probs, discounted_rewards):
            loss -= log_prob * reward
        if(len(episode_log_probs) == 0):
            print(episode_rewards)
        loss /= len(episode_log_probs)
        
        # Update the policy network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the episode rewards
        #print(f"episode {i_episode}: {sum(episode_rewards)}")
        episodes_total_rewards.append(sum(episode_rewards))
        
        
    # Plot the cumulative rewards so far
    plt.plot(episodes_total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards per Episode')
    plt.show()
    
    return policy_network
        