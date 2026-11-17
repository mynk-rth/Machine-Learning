


# action value decided: 0 = LEFT, 1 = RIGHT, 2 = UP

Q = [
    [0, 0, 0],  # State 0
    [0, 0, 0],  # State 1
    [0, 0, 0],  # State 2
    [0, 0, 0]   # State 3 (Goal)
]

alpha = 0.5
gamma = 0.9

# path decided : RIGHT, UP (wall), RIGHT, RIGHT
actions = [1, 2, 1, 1]

state = 0

for action in actions:

    # find next state and reward
 
    if action == 0:        # LEFT
        next_state = max(0, state - 1)
        reward = -1

    elif action == 1:      # RIGHT
        next_state = min(3, state + 1)
        reward = -1

    else:                  # UP = WALL
        next_state = state
        reward = -10

    # goal reward
    if next_state == 3:
        reward = 10

    # old Q (overwrite karna h)
    old_q = Q[state][action]

    # future Q
    if next_state == 3:
        future_q = 0
    else:
        future_q = max(Q[next_state])

    # q-Learning formula
    new_q = old_q + alpha * (
        reward + gamma * future_q - old_q
    )

    # update Q-table
    Q[state][action] = new_q

    print("\nCurrent State:", state)
    print("Action:", action)
    print("Next State:", next_state)
    print("Reward:", reward)
    print("Old Q:", old_q)
    print("Future Q:", future_q)
    print("New Q:", new_q)

    state = next_state

    if state == 3:
        print("Goal Reached!")
        break

print("\nFinal Q-table:")
for i in range(4):
    print("State", i, Q[i])
