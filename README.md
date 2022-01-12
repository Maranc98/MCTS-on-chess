# MCTS on chess
I applied a simple MCTS implementation to the game of chess.

### Prerequisites
The code was tested on python 3.9.
* Gym: `pip install gym`  
* Gym chess: `pip install gym-chess`
* Anytree: `pip install anytree` to build the MCTS tree and display it

### Script
The `mcts.py` script runs the algorithm on 10 rounds of chess. 
This is an example of a script to get the next best move as computed my MCTS:

```python
# Assigns an environment to the algorithm to get legal moves for each iteration
solver = MCTS(gym.make('Chess-v0'))

# Gets the next best move by computing MCTS steps for 10 seconds, for a max of 1000 steps
move = solver.get_best_move(steps=1000, max_time=10)

# Adds the chosen move to the MCTS model to update the new S0 accordingly
solver.add_move(move)
```

### Behaviour
The MCTS algorithm behaves as shown in the following images. 
To better visualize the asymmetric exploration I reduced the exploration factor and only considered two legal moves for each step.
The graph data is saved as `.dot` files in `trees/`. I then manually converted it to images using [graphviz](https://graphviz.org/)

<img src="https://user-images.githubusercontent.com/48620867/149146716-bb85d702-18c2-40a8-9e99-d31bf4082d87.gif" width="250" height="250"/>
![example](https://user-images.githubusercontent.com/48620867/149146716-bb85d702-18c2-40a8-9e99-d31bf4082d87.gif)
