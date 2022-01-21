# MCTS on chess
I applied a simple MCTS implementation to the game of chess.

### Prerequisites
The code was tested on python 3.9.
* Gym: `pip install gym`  
* Gym chess: `pip install gym-chess`
* Anytree: `pip install anytree` to build the MCTS tree and display it
* Pytorch: Go here to learn how to install the correct version for your environment.
* Pytorch-lightning: `pip install pytorch-lightning` if you want to train and use the board evaluation model
* Stockfish: `pip install stockfish` to use the StockfishPlayer, you will also need to add stockfish binaries to its folder

### Script
The `mcts.py` script runs the algorithm on 10 rounds of chess. 
Player1 uses MCTS with a pre-trained CNN model to evaluate board states. The model is trained on [this]!(https://www.kaggle.com/ronakbadhe/chess-evaluations) dataset to predict Stockfish evaluations of a given board state.
Player2 uses Stockfish to choose its moves.
Different simple players and rollout functions are defined in their respective folders. 
They can be used with the defined MCTS interface.

### Example of usage
This is an example of a script to get the next best move as computed my MCTS:

```python
# Defines the MCTS player with your choice of rollout function and parameters
player1 = MCTS(
    rollout_fn=RandomRollout(repeat_n=2),
    max_steps=1000,
    max_time=5
)

# Creates a chess gym environment
env = gym.make('Chess-v0')
env.reset()

# Updates the player with the current board state
player1.set_state(env)

# Computes the optimal move given the current state
move = player1.get_best_move()

# Updates the environment with the chosen move
env.step(move)
```

After the time or steps cap is reached, the algorithm chooses which move to take.
The script in `mcts.py` then makes the move on an external gym-chess environment and prints the resulting board.
The opponent then makes its move, and the script visualizes the new board.
This process can be repeated until game completion.

This is an example output of a MCTS step followed by a step from the opponent.
```
♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ♙ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
♙ ♙ ♙ ♙ ♙ ⭘ ♙ ♙
♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖ 

♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜
♟ ♟ ♟ ♟ ♟ ⭘ ♟ ♟
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ♟ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ♙ ⭘ ⭘
⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘ ⭘
♙ ♙ ♙ ♙ ♙ ⭘ ♙ ♙
♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖
```

### Behaviour
The MCTS algorithm behaves as shown in the following images. 
To better visualize the asymmetric exploration I reduced the exploration factor and only considered two legal moves for each step.
The graph data is saved as `.dot` files in `trees/` through the `save_tree` argument of the `MCTS` object. I then manually converted it to images using [graphviz](https://graphviz.org/).

<img src="https://user-images.githubusercontent.com/48620867/149146716-bb85d702-18c2-40a8-9e99-d31bf4082d87.gif" height="400"/>

