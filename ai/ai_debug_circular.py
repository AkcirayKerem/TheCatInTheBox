from game.agent import Agent
from game.game import Game
from config import MAX_TURNS_PER_GAME

agent = Agent(network_type="circular")

# Preserve the original network
genome = agent.network.get_genome().copy()

timeout_seed = None

print("Searching for timeout...", flush=True)

# Find a timeout using the SAME network
for seed in range(300000, 301000):

    agent.network.set_genome(genome)
    agent.reset(seed)

    game = Game(seed)

    while not game.finished:

        if game.turns >= MAX_TURNS_PER_GAME:
            timeout_seed = seed
            break

        guess = agent.choose_guess()
        found = game.guess(guess)

        if found:
            break

        agent.update(guess)

    if timeout_seed is not None:
        break

if timeout_seed is None:
    print("No timeout found.")
    exit()

print("Timeout seed:", timeout_seed)

# Restore the EXACT same network and reproduce
agent.network.set_genome(genome)
agent.reset(timeout_seed)

game = Game(timeout_seed)

print("\n========== GAME TRACE ==========")

for turn in range(40):

    scores = agent.network.forward(agent.probabilities)

    guess = agent.choose_guess()

    print(f"\nTurn {turn + 1}")
    print("Actual cat position:", game.number)
    print("Beliefs:", agent.probabilities.round(5))
    print("Scores:", scores.round(5))
    print("Guess:", guess)

    found = game.guess(guess)

    if found:
        print("Cat found!")
        break

    agent.update(guess)