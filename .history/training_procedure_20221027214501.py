from helpers import *
from implementations import *
from tqdm import tqdm

# def train_first_feature(y, tx, initial_ws, initial_gamma = 0.001, num_iterations = 10, increase_limit = 0):
#     gamma = initial_gamma
#     sgd_ws = initial_ws
#     losses = []
#     best_loss = 10e10
#     best_w = initial_ws
#     num_increases = 0
#     for i in tqdm(range(num_iterations)):
#         sgd_ws, sgd_losses = least_squares_SGD(y, tx, sgd_ws, 1, gamma)
#         if i > 0 and losses[-1] < sgd_losses:
#             num_increases += 1
#             if num_increases > increase_limit :
#                 gamma = gamma / 2
#         else:
#             num_increases = 0
#             if sgd_losses < best_loss:
#                 best_loss = sgd_losses
#                 best_w = sgd_ws
#         losses.append(sgd_losses)
#     return best_w, sgd_ws, losses

def train_first_feature(y, tx, initial_ws, initial_gamma = 0.001, num_iterations = 10, increase_limit = 0):
    gamma = initial_gamma
    sgd_ws = initial_ws
    losses = []
    best_loss = 10e10
    best_w = initial_ws
    num_increases = 0
    for i in tqdm(range(num_iterations)):
        sgd_ws, sgd_losses = least_squares_SGD(y, tx, sgd_ws, 1, gamma)
        if i > 0 and losses[-1] < sgd_losses:
            num_increases += 1
            if num_increases > increase_limit :
                gamma = gamma / 2
        else:
            num_increases = 0
            if sgd_losses < best_loss:
                best_loss = sgd_losses
                best_w = sgd_ws
        losses.append(sgd_losses)
        print(f"Linear Regression SGD({i}/{num_iterations - 1}): loss={sgd_losses}")
    return best_w, best_loss