from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
from heapdict import *



class BFSAgent():
    def __init__(self) -> None:
        self.env = None

    def calculate_solution(self, env:DragonBallEnv, child:Tuple, expended:int, last_actions:dict, parents:dict, costs:dict) -> Tuple[List[int], float, int]:
      first_state = self.env.get_initial_state()
      actions = []
      total_cost = 0
      curr_state = child
      while curr_state != first_state:
          actions.insert(0,last_actions[curr_state])
          total_cost+=costs[curr_state]
          curr_state = parents[curr_state]
      return actions, total_cost, expended

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()

        curr_state = self.env.get_initial_state()
        if curr_state in self.env.get_goal_states():
          return [],0,0

        open_list = []
        close_list = []
        visited_holes =[]
        expended = 0
        last_actions = {}
        parents = {}
        costs = {}

        open_list.append(curr_state)

        while open_list:
          curr_state = open_list.pop(0)
          expended += 1
          g_state = curr_state[0] in [goal[0] for goal in self.env.get_goal_states()]
          if self.env.succ(curr_state)[0][0] == None or g_state:
            if curr_state in visited_holes: 
              expended -=1
            else:
              visited_holes.append(curr_state)
            continue
          close_list.append(curr_state)
          
          for action in range(4):
            self.env.reset()
            self.env.set_state(curr_state)
            new_state,cost,terminated = self.env.step(action) # child = new_state, cost, terminated
            if new_state not in open_list and new_state not in close_list:
              parents[new_state] = curr_state
              costs[new_state] = cost
              last_actions[new_state] = action

              if self.env.is_final_state(new_state):
                return self.calculate_solution(self.env,new_state, expended, last_actions, parents, costs)
              open_list.append(new_state)
        return [],0,0
		
		
		
		
		
def calc_manhatten_dist(state1:Tuple, state2:Tuple, env:DragonBallEnv):
  row1, col1 = env.to_row_col(state1)
  row2, col2 = env.to_row_col(state2)
  return abs(row1-row2) + abs(col1-col2)

def hmsap_hueristics(state_to_calc:Tuple, env:DragonBallEnv):
  collected =[]
  if not state_to_calc[1]:
    collected += [env.d1]
  if not state_to_calc[2]:
   collected += [env.d2] 
  goals = env.get_goal_states() +collected
  min_dist = 1000000
  for goal in goals:
    curr_dist = calc_manhatten_dist(state_to_calc, goal,env)
    if curr_dist < min_dist:
      min_dist = curr_dist
  return min_dist

  


class WeightedAStarAgent():
    def __init__(self) -> None:
      self.env = None

    def calculate_solution(self, env:DragonBallEnv, child:Tuple, expended:int, last_actions:dict, parents:dict, costs:dict) -> Tuple[List[int], float, int]:
      first_state = self.env.get_initial_state()
      actions = []
      curr_state = child
      while curr_state != first_state:
          actions.insert(0,last_actions[curr_state])
          curr_state = parents[curr_state]
      return actions, costs[child], expended

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
      self.env = env
      self.env.reset()

      curr_state = self.env.get_initial_state()
      if curr_state in self.env.get_goal_states():
        return [],0,0

      open_list = heapdict()
      close_list = []
      visited_holes=[]
      expended = 0
      last_actions = {}
      parents = {}
      cost_until_now = {}
      f_of_start = hmsap_hueristics(curr_state,env) * h_weight
      open_list[curr_state] = (f_of_start,curr_state[0])
      cost_until_now[curr_state]=0

      while open_list:
          curr_state,priority = open_list.popitem() 
          close_list.append(curr_state)
          if self.env.is_final_state(curr_state):
              return self.calculate_solution(self.env,curr_state, expended, last_actions, parents, cost_until_now)
          expended += 1
          if self.env.succ(curr_state)[0][0] == None  or curr_state[0] in [goal[0] for goal in self.env.get_goal_states()]:
              
            continue
         
          for action in range(4):
            self.env.reset()
            self.env.set_state(curr_state)
            child = self.env.step(action) # child = new_state, cost, terminated
            is_new_node = child[0] not in open_list and child[0] not in close_list
            is_better_route = not is_new_node and (cost_until_now[curr_state] + child[1]) < cost_until_now[child[0]] 
            if is_new_node or is_better_route:
              parents[child[0]] = curr_state
              cost_until_now[child[0]] = cost_until_now[curr_state] + child[1]
              last_actions[child[0]] = action
              f_function = hmsap_hueristics(child[0], env)*h_weight + cost_until_now[child[0]]* (1-h_weight)
              open_list[child[0]] = (f_function,child[0][0])
      return [],0,0






class AStarEpsilonAgent():
    def __init__(self) -> None:
      self.env = None
    def calculate_solution(self, env:DragonBallEnv, child:Tuple, expended:int, last_actions:dict, parents:dict, costs:dict) -> Tuple[List[int], float, int]:
      first_state = self.env.get_initial_state()
      actions = []
      curr_state = child
      while curr_state != first_state:
          actions.insert(0,last_actions[curr_state])
          curr_state = parents[curr_state]
      return actions, costs[child], expended

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
      self.env = env
      self.env.reset()

      curr_state = self.env.get_initial_state()
      if curr_state in self.env.get_goal_states():
        return [],0,0

      open_list = heapdict()
      focal = heapdict()
      close_list = []
      visited_holes =[]
      expended = 0
      last_actions = {}
      parents = {}
      cost_until_now = {}

      open_list[curr_state] = (0,0)
      cost_until_now[curr_state]=0

      while open_list:
          curr_state, priority = open_list.peekitem()
          curr_f = priority[0]

          for state, priority in open_list.items():
             if priority[0] <= (1+epsilon) * curr_f:
                focal[state] = cost_until_now[state], state[0]
          
          curr_state, _ = focal.popitem()
          open_list.pop(curr_state)
          close_list.append(curr_state)
          focal.clear()
          if self.env.is_final_state(curr_state):
                return self.calculate_solution(self.env,curr_state, expended, last_actions, parents, cost_until_now)
          expended += 1
          if self.env.succ(curr_state)[0][0] == None or curr_state[0] in [goal[0] for goal in self.env.get_goal_states()]:
            continue
          
          for action in range(4):
            self.env.reset()
            self.env.set_state(curr_state)
            child = self.env.step(action) # child = new_state, cost, terminated
            is_new_node = child[0] not in open_list and child[0] not in close_list
            is_better_route = not is_new_node and (cost_until_now[curr_state] + child[1]) < cost_until_now[child[0]] 
            if is_new_node or is_better_route:
              parents[child[0]] = curr_state
              cost_until_now[child[0]] = cost_until_now[curr_state] + child[1]
              last_actions[child[0]] = action
              f_function = hmsap_hueristics(child[0], env)*0.5 + cost_until_now[child[0]]* 0.5
              open_list[child[0]] = (f_function,child[0][0])
      return [],0,0