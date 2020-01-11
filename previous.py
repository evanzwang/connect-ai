
# Hyper-parameters
NUM_PLAYERS = 2
WIDTH = 7
HEIGHT = 6
CONNECT_NUM = 4

EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 0.01
MEMORY_CAPACITY = 100000
LEARNING_RATE = 0.001
GAMMA = 0.999

BATCH_SIZE = 128
NUM_EPISODES = 300
TARGET_UPDATE = 10
REWARD_ARR = [0.1, 1000, -0.1, -0.1]

# Initializes agent and memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(EpsilonGreedy(EPS_START, EPS_END, EPS_DECAY), WIDTH, device)
memory = ReplayMemory(MEMORY_CAPACITY)

policy_net = DQN(WIDTH, HEIGHT, NUM_PLAYERS, WIDTH).to(device)
target_net = DQN(WIDTH, HEIGHT, NUM_PLAYERS, WIDTH).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=LEARNING_RATE)

env = ConnectEnv(WIDTH, HEIGHT, CONNECT_NUM, NUM_PLAYERS, REWARD_ARR)

# Training processes
for episode in range(NUM_EPISODES):
    env.reset()
    current_player = env.current_player + 1
    state = env.render_perspective(current_player)
    for t in count():
        action = agent.select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        next_state = env.render_perspective(current_player)

        state = next_state
        memory.push(Experience(state, action, next_state, reward))

        if len(memory) >= BATCH_SIZE:
            experiences = memory.sample(BATCH_SIZE)
            batch = Experience(*zip(*experiences))
            states = torch.cat(batch.state)
            actions = torch.cat(batch.action)
            next_states = torch.cat(batch.next_state)
            rewards = torch.cat(batch.reward)

            curr_state_values = policy_net(states).gather(1, actions)
            next_state_values = target_net(next_states).max(1)[0].detach()
            expected_values = (next_state_values * GAMMA) + rewards

            loss = F.mse_loss(curr_state_values, expected_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()

            # May not be necessary
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)

            optimizer.step()

        if done:
            break

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        current_player = env.current_player + 1

print("Done")

# Testing
print(torch.randn(4))
print("test")
print("test2")
