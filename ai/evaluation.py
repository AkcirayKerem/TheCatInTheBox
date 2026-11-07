from config import MAX_TURNS_PER_GAME
from game.game import Game
def play_game(agent, seed=None):

    game = Game(seed)
    agent.reset(seed)

    while not game.finished:

        # Prevent an evaluation from running indefinitely
        if game.turns >= MAX_TURNS_PER_GAME:
            return MAX_TURNS_PER_GAME

        guess = agent.choose_guess()

        found = game.guess(guess)

        if found:
            return game.turns

        agent.update(guess)

    return game.turns
def evaluate_agent(agent, games=100, seed_offset=0):

    total_turns = 0

    for seed in range(seed_offset, seed_offset + games):

        total_turns += play_game(agent, seed)

    return total_turns / games
def evaluate_agent_worker(args):

    agent, games, seed_offset = args

    return evaluate_agent(
        agent,
        games=games,
        seed_offset=seed_offset
    )
def validate_agent(agent, games=1000, seed_offset=100000):
    return evaluate_agent(
        agent,
        games=games,
        seed_offset=seed_offset
    )