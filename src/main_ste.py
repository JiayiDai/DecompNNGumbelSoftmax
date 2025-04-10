import copy
import os, random, torch, tyro, numpy as np
from typing import List
from dataclasses import dataclass
from typing import Union, List
from agent.ppo_agent import PPOAgent, Trajectory
from environment.environments_combogrid_gym import ComboGym
from environment.environments_combogrid import PROBLEM_NAMES as COMBO_PROBLEM_NAMES
from tabulate import tabulate
import torch.autograd as autograd
import torch.nn.functional as F
import matplotlib.pyplot as plt

@dataclass
class Args:
    exp_name: str = "extract_option"
    """the name of this experiment"""
    env_seeds: Union[List[int], str] = (0,1,2,3)
    """seeds used to generate the trained models. It can also specify a closed interval using a string of format 'start,end'."""

    model_paths: List[str] = (
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd0_TL-BR/seed=0',
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd1_TR-BL/seed=0',
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd2_BR-TL/seed=0',
        'train_ppoAgent_ComboGrid_gw5_h64_l10_lr0.00025_clip0.2_ent0.01_envsd3_BL-TR/seed=0',
    )

    # These attributes will be filled in the runtime
    exp_id: str = ""
    """The ID of the finished experiment; to be filled in run time"""
    problems: List[str] = ()
    """the name of the problems the agents were trained on; To be filled in runtime"""

    # Algorithm specific arguments
    env_id: str = "ComboGrid"
    """the id of the environment corresponding to the trained agent
    """
    number_actions: int = 3
    """"Length of the sequence used in experiments."""
    
    # Domain values
    """the width of the puzzles"""
    game_width: int = 5
    """size of observation"""
    observation_length: int = 59

    # hyperparameters
    """the length of the combo/mini grid square"""
    hidden_size: int = 64
    """"""
    l1_lambda: float = 0
    """"""

    # mask learning
    mask_learning_rate: float = 0.001
    """"""
    mask_learning_steps: int = 2000
    """"""
    max_grad_norm: float = 1.0
    """"""
    input_update_frequency: int = 1

    # Script arguments
    seed: int = 0
    """The seed used for reproducibilty of the script"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    grad_est: str = ""
    gumbel_decay: float = 1e-4
    gumbel_t: float = 1
    gumbel_t_init: float = 1

def process_args() -> Args:
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # setting the experiment id
    if args.exp_id == "":
        args.exp_id = f'{args.exp_name}_{args.env_id}' + \
        f'_gw{args.game_width}_h{args.hidden_size}_l1{args.l1_lambda}' + \
        f'_envsd{",".join(map(str, args.env_seeds))}'
    
    # Processing seeds from commands
    if isinstance(args.env_seeds, list) or isinstance(args.env_seeds, tuple):
        args.env_seeds = list(map(int, args.env_seeds))
    elif isinstance(args.env_seeds, str):
        start, end = map(int, args.env_seeds.split(","))
        args.env_seeds = list(range(start, end + 1))
    else:
        raise NotImplementedError
    
    args.problems = [COMBO_PROBLEM_NAMES[seed] for seed in args.env_seeds]

    return args


def get_single_environment(args: Args, seed):
    problem = COMBO_PROBLEM_NAMES[seed]
    env = ComboGym(rows=args.game_width, columns=args.game_width, problem=problem)
    return env


def regenerate_trajectories(args: Args, verbose=False):
    """
    This function loads one trajectory for each problem stored in variable "problems".

    The trajectories are returned as a dictionary, with one entry for each problem. 
    """
    trajectories = {}
    
    for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
        model_path = f'models/{model_directory}/ppo_first_MODEL.pt'
        env = get_single_environment(args, seed=seed)
        
        agent = PPOAgent(env, hidden_size=args.hidden_size)
        
        agent.load_state_dict(torch.load(model_path))

        trajectory = agent.run(env, verbose=verbose)
        trajectories[problem] = trajectory

    return trajectories

class STESoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        max_indices = torch.argmax(x, dim=0, keepdim=True)
        one_hot = torch.zeros_like(x)
        one_hot.scatter_(0, max_indices, 1.0)
        return one_hot

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def save_loss_curve(grad_est, losses, problem_agent=None, actions_to_learn=None):
    steps = list(range(len(losses)))
    steps = [20*i for i in steps]
    plt.figure()
    plt.plot(steps, losses)
    plt.title('Loss curve for {}'.format(grad_est))
    plt.xlabel('Step')
    plt.ylabel('Loss')
    #plt.grid(True)
    plt.savefig('loss_curve_{}_{}_{}.png'.format(grad_est, problem_agent, actions_to_learn))

def train_masks(agent: PPOAgent, trajectory: Trajectory, args: Args, problem_agent=None, actions_to_learn=None):
    mask = torch.nn.Parameter(torch.randn(3, args.hidden_size), requires_grad=True)
    mask_probs = torch.softmax(mask, dim=0)
    mask_discretized = STESoftmax.apply(mask_probs)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([mask], lr=args.mask_learning_rate)
    observations = trajectory.get_state_sequence()

    actions = trajectory.get_action_sequence()
    init_trajectory = agent.run_with_mask(observations, mask_discretized, trajectory.get_length())
    init_loss = loss_fn(torch.stack(init_trajectory.logits), torch.tensor(actions))

    best_mask = mask_discretized
    best_loss = init_loss

    mask_losses = []
    
    for step in range(args.mask_learning_steps):
        mask_probs = torch.softmax(mask, dim=0)
        mask_discretized = STESoftmax.apply(mask_probs)
        new_trajectory = agent.run_with_mask(observations, mask_discretized, trajectory.get_length())
        mask_loss = loss_fn(torch.stack(new_trajectory.logits), torch.tensor(actions))
        
        optimizer.zero_grad()
        mask_loss.backward()
        optimizer.step()

        if step % 20 == 0:
            #print("Iteration {} Loss: {}".format(step, mask_loss))
            mask_losses.append(mask_loss.detach().data)
        if mask_loss < best_loss:
            best_loss = mask_loss
            best_mask = mask_discretized
    save_loss_curve(args.grad_est, mask_losses, problem_agent, actions_to_learn)
    print("args.gumbel_t:", args.gumbel_t)
    args.gumbel_t = args.gumbel_t_init
    print("args.gumbel_t:", args.gumbel_t)
    return best_mask.detach().data, init_loss, best_loss

def generate_test_problems(problem_game_dict: dict):
    problem_tests_dict = {}
    for problem, env in problem_game_dict.items():
        list_problems = []
        for i in range(env._game._rows):
            for j in range(env._game._columns):
                # We do not create a test instance where the initial location is the goal cell
                if env._game._matrix_goal[i][j] == 1:
                    continue

                copy_env = copy.deepcopy(env)
                copy_env._game._matrix_unit[copy_env._game._x][copy_env._game._y] = 0
                copy_env._game._x = i
                copy_env._game._y = j
                copy_env._game._matrix_unit[copy_env._game._x][copy_env._game._y] = 1

                list_problems.append(copy_env)
        problem_tests_dict[problem] = list_problems
    return problem_tests_dict

def evaluate_mask(agent: PPOAgent, mask: torch.nn.Parameter, target_sequence: List, test_problems: dict, length_sequence: int, grad_est: str):
    agent.eval()
    masked_score = 0
    original_score = 0
    total_comparisons = 0

    original_model_mask = torch.zeros_like(mask)
    original_model_mask[-1, :] = 1

    if grad_est == "Gumbel":
        mask = gumbel_sample(mask, training=False)

    for problem, tests in test_problems.items():
        print('Problem: ', problem)
        for test in tests:
            trajectory_masked = agent.run_with_mask(copy.deepcopy(test), mask, length_sequence)
            trajectory_original = agent.run_with_mask(copy.deepcopy(test), original_model_mask, length_sequence)

            actions_masked = trajectory_masked.get_action_sequence()
            actions_original = trajectory_original.get_action_sequence()

            masked_score += sum(1 for x, y in zip(target_sequence, actions_masked) if x == y)
            original_score += sum(1 for x, y in zip(target_sequence, actions_original) if x == y)
            total_comparisons += len(actions_masked)

    return masked_score/total_comparisons, original_score/total_comparisons

def learn_masks(args: Args):
    # Structure to store the trajectories used to train the masks
    sequences_to_learn = {}
    sequences_to_learn[(0, 0, 1)] = Trajectory()
    sequences_to_learn[(0, 1, 2)] = Trajectory()
    sequences_to_learn[(2, 1, 0)] = Trajectory()
    sequences_to_learn[(1, 0, 2)] = Trajectory()
    # Structure to store the best mask, according to the training data; one mask for each segment
    masks_results = {}
    masks_results[(0, 0, 1)] = []
    masks_results[(0, 1, 2)] = []
    masks_results[(2, 1, 0)] = []
    masks_results[(1, 0, 2)] = []

    # Dictionary used to print the results of the masking, where we map a sequence to the semantics of the sequence
    sequence_name = {}
    sequence_name[(0, 0, 1)] = 'Up'
    sequence_name[(0, 1, 2)] = 'Down'
    sequence_name[(2, 1, 0)] = 'Left'
    sequence_name[(1, 0, 2)] = 'Right'

    # Collecting state-action pairs for each segment
    problem_trajectory_dict = regenerate_trajectories(args, verbose=True)
    problem_agent_dict = {}
    problem_game_dict = {}

    for _, trajectory in problem_trajectory_dict.items():
        for i in range(0, trajectory.get_length(), args.number_actions):
            sliced_trajectory = trajectory.slice(i - args.number_actions, i)
            actions = sliced_trajectory.get_action_sequence()
            sequences_to_learn[tuple(actions)].concat(sliced_trajectory)

    # Generating a set of test states, which includes all possible cells of all environments
    for seed, problem, model_directory in zip(args.env_seeds, args.problems, args.model_paths):
        model_path = f'models/{model_directory}/ppo_first_MODEL.pt'
        print(f'Extracting from the agent trained on {problem}, env_seed={seed}')
        env = get_single_environment(args, seed=seed)
        
        agent = PPOAgent(env, hidden_size=args.hidden_size)
        agent.load_state_dict(torch.load(model_path))

        problem_agent_dict[problem] = agent
        problem_game_dict[problem] = copy.deepcopy(env)

    test_problems = generate_test_problems(problem_game_dict)

    # Training the masks
    for problem_agent, agent in problem_agent_dict.items():
        print('Agent trained on: ', problem_agent)
        for actions_to_learn, trajectory in sequences_to_learn.items():
            print('Trying to learn sequence: ', actions_to_learn)

            trained_mask, init_loss, final_loss = train_masks(agent, trajectory, args, problem_agent, actions_to_learn)
            
            print("Initial Loss: ", init_loss, "Final loss: ", final_loss, "Masks Learned: ", trained_mask)

            masked_score, original_score = evaluate_mask(agent, trained_mask, actions_to_learn, test_problems, args.number_actions, args.grad_est)
            masks_results[tuple(actions_to_learn)].append((masked_score, original_score, problem_agent))
            print("Masked Score: ", masked_score, " Original Score: ", original_score)

    
    for sequence, results in masks_results.items():
        data = []
        for result in results:
            masked_score, original_score, problem_agent = result
            data.append([sequence_name[sequence], masked_score, original_score, problem_agent])
        print(tabulate(data, headers=["Sequence", "Masked Score", "Original Score", "Base Model"], tablefmt="grid"))
        print()

def main():
    args = process_args()
    print("grad_est:", args.grad_est)
    learn_masks(args)

if __name__ == "__main__":
    main()