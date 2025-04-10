import numpy as np
import math
import gc


SEEDS = {
    "TL-BR": 0,
    "TR-BL": 1, 
    "BR-TL": 2, 
    "BL-TR": 3,

    "ML-BR": 4,
    "ML-TR": 5,
    "MR-TL": 6,
    "MR-BL": 7,
    "TL-MR": 8,
    "BL-MR": 9,
    "TR-ML": 10,
    "BR-ML": 11,

    "BL-MR-ML-BM-TM": 12,
}

PROBLEM_NAMES = list(SEEDS.keys())

DIRECTIONS = {
(0, 0, 1): "U",
(0, 1, 2): "D",
(2, 1, 0): "L",
(1, 0, 2): "R",
}

class Problem:
    def __init__(self, rows, columns, problem_str):
        self.rows = rows
        self.columns = columns
        self.initial, self.goals = self._parse_problem(problem_str)
        self.reset()

    def _parse_problem(self, problem_str):
        problem_parts = problem_str.split("-")
        initial = self._parse_position(problem_parts[0])
        goals = [self._parse_position(goal_str) for goal_str in problem_parts[1:]]
        return initial, goals

    def _parse_position(self, pos_str):
        v_loc, h_loc = pos_str[0], pos_str[1]
        if v_loc == 'T':
            row = 0
        elif v_loc == 'B':
            row = self.rows - 1
        elif v_loc == 'M':
            row = math.floor(self.rows / 2)
        else:
            raise ValueError("Invalid row specifier. Use 'T' for top, 'B' for bottom, or 'M' for middle.")
        
        if h_loc == 'L':
            col = 0
        elif h_loc == 'R':
            col = self.columns - 1
        elif h_loc == 'M':
            col = math.floor(self.columns / 2)
        else:
            raise ValueError("Invalid column specifier. Use 'L' for left, 'R' for right, or 'M' for middle.")
        
        return (row, col)
    
    def update_goal(self) -> bool:
        """
        Updates the goal coordinations.

        Returns whether or not all goals have been reached.
        
        Returns:
            bool: True if all goals have been reached, False otherwise.
        """

        if self.goal_idx == len(self.goals) - 1:
            return True
        
        self.goal_idx += 1
        self.goal = self.goals[self.goal_idx]
        return False
    
    def reset(self):
        self.goal_idx = 0
        self.goal = self.goals[self.goal_idx]

    def is_goal(self, loc):
        return any([(loc[0] == goal[0]) and (loc[1]==goal[1]) for goal in self.goals])
        


class Game:
    """
    The (0, 0) in the matrices show top and left and it goes to the bottom and right as 
    the indices increases.
    """
    def __init__(self, rows, columns, problem_str, init_x=None, init_y=None):
        self._rows = rows
        self._columns = columns

        self.problem = Problem(rows, columns, problem_str)
        self._matrix_structure = np.zeros((rows, columns))
        
        if init_x and init_y:
            self.reset((init_x, init_y))
        else:
            self.reset()

        # state of current action sequence
        """
        Mapping used: 
        0, 0, 1 -> up (0)
        0, 1, 2 -> down (1)
        2, 1, 0 -> left (2)
        1, 0, 2 -> right (3)
        """
        self._pattern_length = 3

        self._action_pattern = {}
        self._action_pattern[(0, 0, 1)] = 0
        self._action_pattern[(0, 1, 2)] = 1
        self._action_pattern[(2, 1, 0)] = 2
        self._action_pattern[(1, 0, 2)] = 3

    def reset(self, init_loc=None):
        self._matrix_unit = np.zeros((self._rows, self._columns))
        self.problem.reset()
        initial = self.problem.initial
        self._x, self._y = init_loc if init_loc else initial
        self._matrix_unit[self._x][self._y] = 1
        self._state = []
        self.set_goal()   
        gc.collect()

    def set_goal(self):
        self._matrix_goal = np.zeros((self._rows, self._columns))
        goal = self.problem.goal
        self._matrix_goal[goal[0]][goal[1]] = 1

    def __repr__(self) -> str:
        str_map = ""
        for i in range(self._rows):
            for j in range(self._columns):
                if self._matrix_unit[i][j] == 1:
                    str_map += " A "
                elif self._matrix_structure[i][j] == 1:
                     str_map += " B "
                elif self._matrix_goal[i][j] == 1:
                     str_map += " G "
                elif self.problem.is_goal((i,j)):
                     str_map += " g "
                else: 
                     str_map += " 0 "
            str_map += "\n"
        return str_map
    
    def represent_options(self, options: dict) -> str:
        str_map = ""
        option_letters = "UDLR"
        for i in range(self._rows):
            for j in range(self._columns):
                if self._matrix_structure[i][j] == 1:
                     str_map += " B "
                elif self._matrix_goal[i][j] == 1:
                     str_map += " G "
                elif self.problem.is_goal((i,j)):
                     str_map += " g "
                elif (i,j) in options and tuple(options[(i,j)]) in self._action_pattern:
                    str_map += f" {option_letters[self._action_pattern[tuple(options[(i,j)])]]} "
                elif self._matrix_unit[i][j] == 1:
                    str_map += " A "
                else: 
                     str_map += " 0 "
            str_map += "\n"
        return str_map
    
    def get_observation(self):
        one_hot_matrix_state = np.zeros((self._pattern_length, self._pattern_length), dtype=int)
        for i, v in enumerate(self._state):
            one_hot_matrix_state[v][i] = 1
        return np.concatenate((self._matrix_unit.ravel(), one_hot_matrix_state.ravel(), self._matrix_goal.ravel()))
    
    def is_over(self):
        if self._matrix_goal[self._x][self._y] == 1:
            done = self.problem.update_goal()
            self.set_goal()
            return done
        return False
    
    def get_actions(self):
        return [0, 1, 2]
   
    def apply_action(self, action):
        """
        Mapping used: 
        0, 0, 1 -> up (0)
        0, 1, 2 -> down (1)
        2, 1, 0 -> left (2)
        1, 0, 2 -> right (3)
        """
        # each column in _state_matrix represents an action
        self._state.append(action)

        if len(self._state) == self._pattern_length:
            action_tuple = tuple(self._state)
            if action_tuple in self._action_pattern:
                # moving up
                if self._action_pattern[action_tuple] == 0:
                    if self._x - 1 >= 0 and self._matrix_structure[self._x - 1][self._y] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._x -= 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving down
                if self._action_pattern[action_tuple] == 1:
                    if self._x + 1 < self._matrix_unit.shape[0] and self._matrix_structure[self._x + 1][self._y] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._x += 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving left
                if self._action_pattern[action_tuple] == 2:
                    if self._y - 1 >= 0 and self._matrix_structure[self._x][self._y - 1] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._y -= 1
                        self._matrix_unit[self._x][self._y] = 1
                # moving right
                if self._action_pattern[action_tuple] == 3:
                    if self._y + 1 < self._matrix_unit.shape[1] and self._matrix_structure[self._x][self._y + 1] == 0:
                        self._matrix_unit[self._x][self._y] = 0
                        self._y += 1
                        self._matrix_unit[self._x][self._y] = 1
            self._state = []
