import os #For the operating System
import random
import numpy as np #For arrays an mathematics
import torch #Train our data using pytourch
import torch.nn as nn #The Neural Network
import torch.optim as optim #Optimizer
import torch.nn.functional as F #Functional
import torch.autograd as autograd #For stochastic gradient descent
from torch.autograd import Variable
from collections import deque, namedtuple



# Define the DQN Algorithm Parameters
learning_rate = 0.001
discount_factor = 0.99
exploration_prob = 1.0
exploration_decay = 0.995
min_exploration_prob = 0.1
number_of_actions = 20

class Network(nn.Module): # We inherit nn.module

  #State_size represents all observable spaces (Xcoords, Ycoords, Xvelocity, Yvelocity, angle, angularVelocity, BooleanStateRightLeg, BooleanStateLeftLeg) OBSERVATION SHAPE
  #4 Actions [0: do nothing, 1: fire left orientation engine, 2: fire main engine, 3: fire right orientation engine]

    def __init__(self, state_size, action_size, seed = 42): #Default Seed 42
        super(Network, self).__init__() #Activates the inheritance
        self.seed = torch.manual_seed(seed) #Sets the seed

        #Connecting Layers in Nerural Layers
        #Fist connection between the input layer and the first fully connected layer

        #First argument is the number of neurons in the input layer
        # The second argument is the number of neurons in the first fully connected layer
        # Note: this is a userdefined value you can do manual test to check the optimal number of connections in our case 64 was found to be the most optimal
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        #Since this is the final part of the neural network we are taking the neaurons of FC2 and creating a layer with the total actionsize based on probability
        self.fc3 = nn.Linear(64, action_size)


    def forward(self, state): #forward propogate layers

        # These first two lines of code propogate the signal from the input layer to the first layer

        x = self.fc1(state)
        #Rectifier Activation Function
        x =  F.relu(x)
        # returns the signal from the first layer to the 2nd layer
        x = self.fc2(x)
        x = F.relu(x)

        return self.fc3(x)

 
# DEFINE THIS WITHIN THE ENVIRONMENT 

#Create an observation space
#state_shape = env.observation_space.shape #the observation space
#state_size =env.observation_space.shape[0] #the size of the observation space where .shape is a matrix function that gets sizes
# number_actions = env.action_space.n #the number of actions should be 4
#print('state shape: ', state_shape)
#print('state size: ', state_size)
# print('number of actions: ', number_actions)


#Hyper Parameters
learning_rate = 5e-4 # .00005 
minibatch_size = 100
discount_factor = 0.99 #gamma NOTE: close to 1 provides better decision making for future rewards
replay_buffer_size = int(1e5) # contains 100,000 experiences stored in the memory of the AI to stablalize Training
interpolation_parameter = 1e-3 #tau 0.0001

#Experience Replay
#Replay Memory
class ReplayMemory():

  def __init__(self, capacity):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #IF cuda by tourch is available we use the GUP else we use the cpu'
    self.capacity = capacity # the maximum size of the memory buffer
    self.memory = [] #List that will contain the experinces i.e state, action, reward next

  # method that will add experiences while also checking that the defined maximum is always met
  def push(self, event):
    self.memory.append(event) # appends to our memory list
    if len(self.memory) > self.capacity: # If capacity is exceeded remove the first input
      del self.memory[0]


  # Batch Size the number of experiences sampled in the batch
  def sample(self, batch_size): #Randomly selects a batch of experiences
    experiences = random.sample(self.memory, k = batch_size)

    #np.vstack creates a vertical array stack of values
    states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device) #We are converting these stack of states into pytorch tensors
    actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
    rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
    next_state = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
    dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
    return states, next_state, actions, rewards, dones

#DQN Agent Implementation
class Agent():
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate) # Optimizers
        self.memory = ReplayMemory(replay_buffer_size) # from replay buffer size has 100,000 memory
        self.t_step = 0 # Time step for learning maybe defined by the environment


    # This method decides to store expereinces and decides when to learn from them
    # Decompose experience into its separate values
    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state,done)) # stores and decides when to learn from
        #Store its experiences in the replay memory

        #Increments the time step counter
        #Checks every 4 steps
        self.t_step = (self.t_step + 1) % 4

        if self.t_step == 0:
        # When we learn we learn through mini batches
        # We check if the number of experience in the memory is already larger than the minibatches
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(100)
                self.learn(experiences, discount_factor)



    #Function for selecting action based on a fiven state
    #We are using an epsilon greedy action process
    def act(self, state, epsilon = 0.):

        # Note: We need to add an extra dimensoion basedo n the batch
        # We need to that for each state which batch does it belong to

        # unsqueeze helps us define the dimenson of the batch, by setting it to zero the first dimension of our first batch will be at the beginning
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) # converts it into a torch tensor

        self.local_qnetwork.eval() # Set Q network in evalutation mode

        # Checks if torch is eval mode
        with torch.no_grad(): # Makes sure that any fradient computation is disabled
            action_values = self.local_qnetwork(state) # Gets the actions

        # Local qnetwork is an instance of an agent class and agent class inheirts from the module class of NN module we can use the NN mode
        self.local_qnetwork.train()

        if random.random() > epsilon: # return the value with the highest Qvalue

        #Generates a random value from 1 to 0
        #if epsilon is .10, then 10% the bot will prioritze exploration
        #The idea is that if the value exceed epsilon, proceed to choose the highest optimal solution
        #Otherwise decide to choose a random value for exploration

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size)) # selects a random function


    def learn(self, experiences, discount_factor): #Discount Factor (Gamma) Represents the decay of learning / optimality
        states, next_states, actions, rewards, dones = experiences

        # Get max Predicted Q Values
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1) # Forward propogate of next State
        #The detach function actually detaches the resulting tensor from the computation graph, meaning that we wont be tracking gradients for this tensor during the backward propogration
        #We need the maximum value from the direciton 1

        q_targets = rewards + (discount_factor * next_q_targets * (1-dones))
        q_expected = self.local_qnetwork(states).gather(1, actions) # We are gathering all the respective Q values gathered by our network

        #Computing the loss within expected and target Q-values
        loss = F.mse_loss(q_expected, q_targets)

        #We need to backpropagate this loss in order to update the model parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() #perfroms exactly a single optimzation step
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)


    def soft_update(self, local_model, target_model, interpolation_parameter):
        # Loop through the parameters of the target and the local q-network
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)



def Train(agent, env):
    # Initialzing the training parameters
    #The maximum number of episodes of which we want to train our agent
    number_episodes = 2000

    #Maximum Number of time steps per episode
    #We dont want our AI to try too hard
    maximum_number_timesteps_per_episode = 1000

    #Hyper parameters related to our epsilon greedy action policy
    epsilon_starting_value = 1.0
    epsilon_ending_value = 0.01 # the last epsilon value we want to check
    epsilon_decay_value = 0.995
    epsilon = epsilon_starting_value
    scores_on_100_episodes = deque(maxlen = 100)

    #training loop
    for episode in range(1, number_episodes + 1):
        #The first step is to reset the environemtent to its initial state
        state, _ = env.reset()
        score = 0

        #Time steps
        for t in range(maximum_number_timesteps_per_episode):
            action = agent.act(state, epsilon)

            #Rewards is based on the gymnasium
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_on_100_episodes.append(score)
        epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "") # Prints points
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))

        #Checks if the mean of the scores of 100 episodes is larger than 200, we can say that we win
        if np.mean(scores_on_100_episodes) >= 200.0:
            print('\nEnvironment solved in {:d}\tAverage Score: {:2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
            torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
            break
