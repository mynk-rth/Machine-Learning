

# decide the action here
RIGHT = 0
DOWN = 1

# decide the states
S0 = 0
S1 = 1
S2 = 2
S3 = 3
GOAL = 4

# create Q table according the states and the action..
# there are  5 states, 2 actions
# decide the values for 
#  right and down action column 0 = RIGHT, column 1 = DOWN
Q = [
    [0.0, 0.0],  # S0
    [0.0, 0.0],  # S1
    [0.0, 0.0],  # S2
    [0.0, 0.0],  # S3
    [0.0, 0.0]   # GOAL
]

# this is learning rate
alpha = 0.5
# and this is discounted value
gamma = 0.9

# now create action according to the path specified
actions = [RIGHT, DOWN, RIGHT, RIGHT, RIGHT]

state = S0

print("step-by-step SARSA\n")

for step, action in enumerate(actions):

    # find next state and reward
    if state == S0 and action == RIGHT:
        next_state = S0       # there is wall: stay at S0
        reward = -10
        print("S0 --RIGHT--> WALL")

    elif state == S0 and action == DOWN:
        next_state = S1
        reward = -1
        print("S0 --DOWN--> S1")

    elif state == S1 and action == RIGHT:
        next_state = S2
        reward = -1
        print("S1 --RIGHT--> S2")

    elif state == S2 and action == RIGHT:
        next_state = S3
        reward = -1
        print("S2 --RIGHT--> S3")

    elif state == S3 and action == RIGHT:
        next_state = GOAL
        reward = 10
        print("S3 --RIGHT--> GOAL")

    else:
        print("Invalid action")
        break

    old_q = Q[state][action]

    # SARSA update
    if next_state == GOAL:
        future_q = 0.0
    else:
        # actual next action selected by the sequence
        next_action = actions[step + 1]
        future_q = Q[next_state][next_action]

    new_q = old_q + alpha * (
        reward + gamma * future_q - old_q
    )

    Q[state][action] = new_q

    print("Reward:", reward)
    print("Old Q-value:", old_q)
    print("Future Q-value:", future_q)
    print("New Q-value:", round(new_q, 2))
    print("-" * 30)

    state = next_state

    if state == GOAL:
        print("Goal reached")
        break

print("\nFinal Q-table")
print("State   RIGHT   DOWN")

for i, row in enumerate(Q):
    print(f"{i:5}   {row[0]:5.1f}   {row[1]:5.1f}")
