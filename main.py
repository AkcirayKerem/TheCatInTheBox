
import os
from ai.evaluation import validate_agent
from ai.genetic_algorithm import GeneticAlgorithm
from ai.evaluation import evaluate_agent
from game.agent import Agent


def demonstrate_agent(agent, games=3):

    from game.game import Game

    for game_number in range(games):

        print(f"\n========== GAME {game_number + 1} ==========")

        seed = 12345 + game_number

        game = Game(seed=seed)
        agent.reset(seed=seed + 123456789)

        while not game.finished:

            agent.explain_strategy()

            guess = agent.choose_guess()
            found = game.guess(guess)

            print(f"\nGuess: Box {guess}")

            if found:
                print(f"Cat found in {game.turns} guesses!")
                break

            agent.update(guess)


def main():

    evolution = GeneticAlgorithm(
        population_size=100,
        elite_count=10,
        mutation_rate=0.05,
        mutation_strength=0.1,
        games_per_agent=100,
        workers=10
    )

    best_agent, best_average = evolution.run(
        generations=10000
    )

    if best_agent is not None:

        print("\n========== FINAL AI REPORT ==========")

        # Test baseline agent
        baseline_test_average = validate_agent(
            evolution.baseline_agent,
            games=10000,
            seed_offset=200000
        )

        print(
            f"Baseline unseen test average: "
            f"{baseline_test_average:.4f}"
        )

        # Test evolved neural network
        neural_test_average = validate_agent(
            best_agent,
            games=10000,
            seed_offset=200000
        )

        print(
            f"Neural network unseen test average: "
            f"{neural_test_average:.4f}"
        )

        #save_agent(best_agent)
        demonstrate_agent(best_agent, games=3)


if __name__ == "__main__":
    main()
