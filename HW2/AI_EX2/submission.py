from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance, Robot
import random
import math


# def find_closest_package(env: WarehouseEnv, robot: Robot):
#     package1,package2 = env.packages[0], env.packages[1]
#     dist1,dist2 = manhattan_distance(robot.position, package1.position) ,manhattan_distance(robot.position, package2.position)
#     closest_package_dist = min(dist1,dist2)
#     closest_package = package1 if dist1 <= dist2 else package2
#     return closest_package_dist,closest_package


def find_max_profit_package(env: WarehouseEnv, robot: Robot):
    package1, package2 = env.packages[0], env.packages[1]
    dist1, dist2 = manhattan_distance(robot.position, package1.position), manhattan_distance(robot.position,
                                                                                             package2.position)
    profit1, profit2 = 2 * manhattan_distance(package1.position, package1.destination), 2 * manhattan_distance(
        package2.position, package2.destination)
    max_profit_package = package1 if profit1 >= profit2 else package2
    max_profit_dist = dist1 if profit1 >= profit2 else dist2
    return max_profit_dist, max_profit_package


def find_closest_charger(env: WarehouseEnv, robot: Robot):
    charger1, charger2 = env.charge_stations[0], env.charge_stations[1]
    dist1, dist2 = manhattan_distance(robot.position, charger1.position), manhattan_distance(robot.position,
                                                                                             charger2.position)
    closest_charger_dist = min(dist1, dist2)
    closest_charger = charger1 if dist1 <= dist2 else charger2
    return closest_charger_dist, closest_charger


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)

    max_profit_package_dist, max_profit_package = find_max_profit_package(env, robot)
    closest_charger_dist, closest_charger = find_closest_charger(env, robot)

    if robot.package:  # we have a package
        if robot.battery > manhattan_distance(robot.position,
                                              robot.package.destination):  # there is enough battery to go to dest - go there
            heuristic = manhattan_distance(robot.package.position, robot.package.destination) - \
                        manhattan_distance(robot.position, robot.package.destination) + 20 * robot.credit + 50
        else:  # go to charger
            heuristic = 20 * robot.battery + 300 * robot.credit - closest_charger_dist

    else:  # we dont have a package
        if max_profit_package_dist + manhattan_distance(max_profit_package.position,
                                                        max_profit_package.destination) + 1 < robot.battery:  # there is enough battery to pickup a package and go to dest - do that
            heuristic = 300 * robot.credit - max_profit_package_dist
        else:  # go to charge
            heuristic = 300 * (robot.credit + robot.battery) - closest_charger_dist
    return heuristic


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def minimax_heuristics(self, env: WarehouseEnv, agent_id: int, is_my_turn: bool, time_limit: float):
        if time_limit <= 0 or env.done():
            return self.heuristic(env, agent_id)

        # now we progress similar to run_step, but for each turn either max or min version
        operators = env.get_legal_operators(agent_id) if is_my_turn else env.get_legal_operators(not agent_id)
        children = [env.clone() for _ in operators]

        if is_my_turn:  # this is the max variation
            curr_max = -math.inf
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v = self.minimax_heuristics(child, agent_id, not is_my_turn, time_limit - 1.0)
                curr_max = max(curr_max, v)
            return curr_max

        else:  # this is the min version
            curr_min = math.inf
            for child, op in zip(children, operators):
                child.apply_operator(not agent_id, op)
                v = self.minimax_heuristics(child, agent_id, not is_my_turn, time_limit - 1.0)
                curr_min = min(curr_min, v)
            return curr_min

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):  # inspired by the greedyAgent's run_step
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        is_my_turn = True
        children_heuristics = [self.minimax_heuristics(child, agent_id, is_my_turn, time_limit - 1.0) for child in
                               children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def minimax_ab_heuristics(self, env: WarehouseEnv, agent_id: int, is_my_turn: bool, time_limit: float, alpha: float,
                              beta: float):
        if time_limit <= 0 or env.done():
            return self.heuristic(env, agent_id)

        # now we progress similar to run_step, but for each turn either max or min version
        operators = env.get_legal_operators(agent_id) if is_my_turn else env.get_legal_operators(not agent_id)
        children = [env.clone() for _ in operators]

        if is_my_turn:  # this is the max variation
            curr_max = -math.inf
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v = self.minimax_ab_heuristics(child, agent_id, not is_my_turn, time_limit - 1.0, alpha, beta)
                curr_max = max(curr_max, v)
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return math.inf
            return curr_max

        else:  # this is the min version
            curr_min = math.inf
            for child, op in zip(children, operators):
                child.apply_operator(not agent_id, op)
                v = self.minimax_ab_heuristics(child, agent_id, not is_my_turn, time_limit - 1.0, alpha, beta)
                curr_min = min(curr_min, v)
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    return -math.inf
            return curr_min

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):  # inspired by the greedyAgent's run_step
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        is_my_turn = True
        children_heuristics = [
            self.minimax_ab_heuristics(child, agent_id, is_my_turn, time_limit - 1.0, -math.inf, math.inf) for child in
            children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


class AgentExpectimax(Agent):
    # TODO: section d : 1

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def expectimax_heuristics(self, env: WarehouseEnv, agent_id: int, is_my_turn: bool, time_limit: float):
        if time_limit <= 0 or env.done():
            return self.heuristic(env, agent_id)

        operators = env.get_legal_operators(agent_id) if is_my_turn else env.get_legal_operators(not agent_id)
        children = [env.clone() for _ in operators]

        if is_my_turn:
            curr_max = -math.inf
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
                v = self.expectimax_heuristics(child, agent_id, not is_my_turn, time_limit - 1.0)
                curr_max = max(curr_max, v)
            return curr_max

        else:
            curr_min = math.inf
            num_children = 0
            total_v = 0.0

            for child, op in zip(children, operators):
                child.apply_operator(not agent_id, op)
                v = self.expectimax_heuristics(child, agent_id, not is_my_turn, time_limit - 1.0)
                if op == 'move east' or op == 'pick up':
                    v = v*2
                    num_children += 1 #raise num childern since there is more probability to do this action
                num_children += 1 #deafult actions
                total_v += v
            curr_v = total_v/num_children
            return curr_v


    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        is_my_turn = True
        children_heuristics = [self.expectimax_heuristics(child, agent_id, is_my_turn, time_limit - 1.0) for child in
                               children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
