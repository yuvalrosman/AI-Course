from copy import deepcopy
import numpy as np
import copy

def calc_max_sigma(mdp,row,col,U):
    if (row,col) in mdp.terminal_states:
        return 0, None,None
    max = -np.inf
    policy_action = None
    sigma_per_action = np.zeros(4).astype(np.float)
    for i,action in enumerate(['UP','DOWN','RIGHT','LEFT']):
        p_vec = mdp.transition_function[action]
        sum =0
        for index, action_taken in enumerate(['UP', 'DOWN', 'RIGHT', 'LEFT']):
            new_row,new_col = mdp.step((row,col),action_taken)
            sum += float(p_vec[index]) * float(U[new_row][new_col])
        sigma_per_action[i] = sum
        if sum > max:
            max = sum
            policy_action = action
    return max, policy_action, sigma_per_action


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    U = copy.deepcopy(U_init)
    U_tag = copy.deepcopy(U_init)
    iteration =0
    while iteration <1000:
        U = copy.deepcopy(U_tag)
        delta = 0
        for s in range(mdp.num_row * mdp.num_col):
            row,col = int(s//(mdp.num_col)), int(s% (mdp.num_col))
            reward = mdp.board[row][col]
            if reward == 'WALL':
                U_tag[row][col] = None
                continue
            max_sigma,_,__ = calc_max_sigma(mdp,row,col,U_tag)
            U_tag[row][col] = float(reward) + mdp.gamma * max_sigma
            delta = max(np.abs(U_tag[row][col] - U[row][col]),delta)
        if delta < (epsilon* (1-mdp.gamma))/(mdp.gamma): break
        iteration+=1
    return np.array(U).astype(float)



    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    policy = [[None] * mdp.num_col for _ in range(mdp.num_row)]
    for s in range(mdp.num_row * mdp.num_col):
        row,col = int(s//(mdp.num_col)), int(s% (mdp.num_col))
        reward = mdp.board[row][col]
        if (row,col) in mdp.terminal_states or reward == 'WALL': continue
        _,policy[row][col],__ = calc_max_sigma(mdp,row,col,U)
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    P = [[0 for i in range(mdp.num_col * mdp.num_row)] for i in range(mdp.num_row * mdp.num_col)]
    R = [0 for i in range(mdp.num_col*mdp.num_row)]
    U = [[0 for i in range(mdp.num_col)] for i in range(mdp.num_row)]

    ends_and_walls=[]
    for s in range(mdp.num_row*mdp.num_col):
        row,col = int(s//(mdp.num_col)), int(s% (mdp.num_col))
        reward = mdp.board[row][col]
        if (row,col) in mdp.terminal_states:
            R[s] = float(reward)
            ends_and_walls.append(s)
            continue
        elif reward == 'WALL':
            ends_and_walls.append(s)
            continue
        else:
            P[s][s] +=1
            R[s] = float(reward)
            for index,action in enumerate(['UP', 'DOWN', 'RIGHT', 'LEFT']):
                new_row,new_col = mdp.step((row,col),action)
                new_reward = mdp.board[new_row][new_col]
                last_action = policy[row][col]
                if (new_row,new_col) in mdp.terminal_states:
                    R[s] += float(new_reward) * (mdp.gamma * float((mdp.transition_function[last_action][index])))
                elif new_reward == 'WALL':
                    continue
                else:
                    P[s][new_col + mdp.num_col * new_row] -= (mdp.gamma * float((mdp.transition_function[last_action][index])))

    for end_or_wall in reversed(sorted(ends_and_walls)):
        P = np.delete(P, end_or_wall, axis =1)
        P = np.delete(P, end_or_wall, axis =0)
        R = np.delete(R, end_or_wall, axis =0)
    solution = np.linalg.solve(P,R)

    for end_or_wall in ends_and_walls:
        solution = np.insert(solution,end_or_wall,0)

    for s in range(mdp.num_row*mdp.num_col):
        row,col = int(s//(mdp.num_col)), int(s% (mdp.num_col))
        reward = mdp.board[row][col]
        if (row,col) in mdp.terminal_states:
            U[row][col] = reward
        elif reward == 'WALL':
            U[row][col] == None
        else:
            U[row][col] = solution[s]
    return np.array(U).astype(float)


    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    U = [[0 for i in range(mdp.num_col)] for i in range(mdp.num_row)]
    iteration =0
    while iteration <1000:
        unchanged = True
        U = policy_evaluation(mdp,policy_init)
        for s in range(mdp.num_row * mdp.num_col):
            row,col = int(s//(mdp.num_col)), int(s% (mdp.num_col))
            reward = mdp.board[row][col]
            if (row,col) in mdp.terminal_states or reward == 'WALL':
                continue
            max,policy,_ =calc_max_sigma(mdp,row,col,U)
            p_vec = mdp.transition_function[policy_init[row][col]]
            sum =0
            for index,action in enumerate(['UP', 'DOWN', 'RIGHT', 'LEFT']):
                new_row,new_col = mdp.step((row, col), action)
                sum += float(p_vec[index]) * float(U[new_row][new_col])
            if max > sum:
                policy_init[row][col] = policy
                unchanged = False
        if unchanged:
            break
        iteration +=1
    return policy_init
    # ========================




def get_all_policies(mdp, U, epsilon=10**(-3),return_policies=False):  # You can add more input parameters as needed
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    # return_policies: if set to True, returning the policies themselves and not the number of policies
    # return: the number of different policies

    # ====== YOUR CODE: ======
    # Initialize policies as a list of lists with the same shape as mdp.board
    policies = [["" for _ in range(len(mdp.board[0]))] for _ in range(len(mdp.board))]
    for s in range(mdp.num_row * mdp.num_col):
        row, col = int(s // (mdp.num_col)), int(s % (mdp.num_col))
        reward = mdp.board[row][col]
        if (row, col) in mdp.terminal_states or reward == 'WALL':
            continue
        max_sigma_, _, all_sigmas = calc_max_sigma(mdp, row, col, U)
        max_sigma = np.around(max_sigma_,int(-np.log10(epsilon))+1)
        valid_actions = (np.where(max_sigma - np.around(all_sigmas, int(-np.log10(epsilon))+1)   < epsilon)[0]).astype(np.int)
        policies[row][col] = ''.join([["U", "D", "R", "L"][i] for i in valid_actions])
    if not return_policies:
        mdp.print_policy(policies)
    vector_func = np.vectorize(len)
    lengths_vector = vector_func(policies)
    lengths_vector[lengths_vector == 0] = 1
    return np.prod(lengths_vector) if not return_policies else policies

def get_policy_for_different_rewards(mdp,epsilon=10**(-3)):  # You can add more input parameters as needed

    # ====== YOUR CODE: ======
    # Extract terminal states and walls from the original mdp
    terminal_states = mdp.terminal_states
    walls = np.where(np.array(mdp.board) == "WALL")

    # Create a new board with all states initialized to a default reward value
    new_board = np.zeros(np.array(mdp.board).shape, dtype=object)

    # Set the terminal states and walls in the new board
    for point in terminal_states:
        new_board[point[0]][point[1]] = mdp.board[point[0]][point[1]]
    for point in zip(*walls):
        new_board[point[0]][point[1]] = "WALL"

    # Initialize rewards, boards, and policies
    min_reward = -5
    max_reward = 5
    rewards = np.around(np.arange(min_reward, max_reward, 0.01), 2)
    boards = []
    policies = np.empty(len(rewards), dtype=object)
    policy_init = np.full(new_board.shape, 'UP', dtype=object)

    # Set the initial policy for terminal states and walls to None
    for point, _ in np.ndenumerate(new_board):
        if point in terminal_states or new_board[point] == "WALL":
            policy_init[point] = None


    # Create a new board for each reward value and calculate the policy for each board
    for index, reward in enumerate(rewards):
        boards.append(deepcopy(new_board))
        for s in range(mdp.num_row * mdp.num_col):
            row, col = int(s // (mdp.num_col)), int(s % (mdp.num_col))
            if (row, col) in mdp.terminal_states or new_board[(row,col)] == "WALL":
                continue
            boards[index][(row,col)] = reward
        mdp_copy = deepcopy(mdp)
        mdp_copy.board = boards[index].tolist()
        utilities = policy_evaluation(mdp_copy, policy_iteration(mdp_copy, policy_init))
        policies[index] = get_all_policies(mdp_copy, utilities, epsilon,True)

    # Identify the indices where the policy changes and print the policy for each range of rewards
    merged_ranges = [i for i in range(1,len(policies)) if not np.array_equal(policies[i], policies[i - 1])]
    # Print all
    print(f" {min_reward} <= R(s) < {rewards[merged_ranges[0]]} ")
    mdp.print_policy(policies[0])

    for i, j in zip(merged_ranges[:-1], merged_ranges[1:]):
        print(f" {rewards[i]} < R(s) < {rewards[j]} ")
        mdp.print_policy(policies[i + 1])

    print(f" {rewards[merged_ranges[-1]]} < R(s) <= {max_reward} ")
    mdp.print_policy(policies[-1])


    return [rewards[i] for i in merged_ranges]