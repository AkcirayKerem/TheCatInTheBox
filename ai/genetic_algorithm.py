import random
import numpy as np
import os
from ai.evaluation import validate_agent
from concurrent.futures import ProcessPoolExecutor, as_completed
from ai.baseline_agent import BaselineAgent
from game.agent import Agent
from ai.evaluation import evaluate_agent_worker

class GeneticAlgorithm:

    def __init__(
            self,
            network_type="circular",
            population_size=150,
            elite_count=8,
            mutation_rate=0.05,
            mutation_strength=0.1,
            games_per_agent=100,
            tournament_size=3,
            workers=None


    ):
        self.normal_mutation_rate = mutation_rate
        self.normal_mutation_strength = mutation_strength


        self.workers = workers or os.cpu_count() or 1

        self.population_size = population_size
        self.baseline_agent = BaselineAgent()
        self.elite_count = elite_count

        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

        self.games_per_agent = games_per_agent
        self.tournament_size = tournament_size
        self.stagnation_counter = 0

        # Create the initial population
        self.network_type = network_type

        self.population = [
            Agent(network_type=self.network_type)
            for _ in range(population_size)
        ]

        self.baseline_agent = BaselineAgent()

    def save_checkpoint(self, genome, validation_average):
        np.savez(
            "best_checkpoint.npz",
            genome=genome,
            validation_average=validation_average
        )
    def evaluate_agents_batch(
            self,
            executor,
            agents,
            games,
            seed_offset,
            stage_name
    ):
        tasks = [
            (agent, games, seed_offset)
            for agent in agents
        ]

        futures = {
            executor.submit(evaluate_agent_worker, task): agent
            for agent, task in zip(agents, tasks)
        }

        results = []
        completed = 0

        print(
            f"\n{stage_name}: Evaluating {len(agents)} agents "
            f"over {games} games each...",
            flush=True
        )

        for future in as_completed(futures):
            agent = futures[future]
            average = future.result()

            results.append((average, agent))
            completed += 1

            # Only print every 5 agents or when complete to prevent I/O lag
            if completed % 5 == 0 or completed == len(agents):
                print(
                    f"\rAgents evaluated: {completed}/{len(agents)} "
                    f"({completed / len(agents) * 100:.1f}%)",
                    end="",
                    flush=True
                )
        print("\nEvaluation complete!", flush=True)

        results.sort(key=lambda x: x[0])
        return results

    def evaluate_population(self, executor, seed_offset):

        # Evaluation configuration
        initial_games = 25
        second_games = 50
        final_games = 75

        first_survivors = 50    
        final_survivors = 25
        # ==========================================
        # STAGE 1: Evaluate the entire population
        # ==========================================

        stage1_results = self.evaluate_agents_batch(
            executor,
            self.population,
            initial_games,
            seed_offset,
            "STAGE 1"
        )

        survivors = [
            agent for _, agent in stage1_results[:first_survivors]
        ]

        previous_averages = {
            id(agent): average
            for average, agent in stage1_results[:first_survivors]
        }

        print(f"Stage 1 survivors: {len(survivors)}")

        # ==========================================
        # STAGE 2: Evaluate the strongest 60
        # ==========================================

        stage2_results = self.evaluate_agents_batch(
            executor,
            survivors,
            second_games,
            seed_offset + initial_games,
            "STAGE 2"
        )

        cumulative_results = []

        for new_average, agent in stage2_results:
            old_average = previous_averages[id(agent)]

            cumulative_average = (
                                         old_average * initial_games
                                         + new_average * second_games
                                 ) / (initial_games + second_games)

            cumulative_results.append(
                (cumulative_average, agent)
            )

        cumulative_results.sort(key=lambda x: x[0])

        survivors = [
            agent for _, agent in cumulative_results[:final_survivors]
        ]

        previous_averages = {
            id(agent): average
            for average, agent in cumulative_results[:final_survivors]
        }

        print(f"Stage 2 survivors: {len(survivors)}")

        # ==========================================
        # STAGE 3: Final evaluation of 40 candidates
        # ==========================================

        stage3_results = self.evaluate_agents_batch(
            executor,
            survivors,
            final_games,
            seed_offset + initial_games + second_games,
            "STAGE 3"
        )

        final_results = []

        for new_average, agent in stage3_results:
            old_average = previous_averages[id(agent)]

            cumulative_average = (
                                         old_average * (initial_games + second_games)
                                         + new_average * final_games
                                 ) / (
                                         initial_games + second_games + final_games
                                 )

            final_results.append(
                (cumulative_average, agent)
            )

        final_results.sort(key=lambda x: x[0])

        print("\n========== SUCCESSIVE EVALUATION ==========")
        print(f"Initial population: {len(self.population)}")
        print(f"Final candidates: {len(final_results)}")
        print(
            f"Games per finalist: "
            f"{initial_games + second_games + final_games}"
        )
        print(
            f"Total games: "
            f"{len(self.population) * initial_games + first_survivors * second_games + final_survivors * final_games}"
        )

        return final_results

    def select_parent(self, results, exclude=None):

        candidates = [
            result for result in results
            if result[1] is not exclude
        ]

        contestants = random.sample(
            candidates,
            self.tournament_size
        )

        winner = min(contestants, key=lambda x: x[0])

        return winner[1].network.get_genome().copy()

    def calculate_diversity(self):

        genomes = np.array([
            agent.network.get_genome()
            for agent in self.population
        ])

        gene_std = np.std(genomes, axis=0)

        return np.mean(gene_std)

    def crossover(self, parent1, parent2):

        mask = np.random.rand(len(parent1)) < 0.5

        child = np.where(mask, parent1, parent2)

        return child

    def mutate(self, genome):

        mask = np.random.rand(len(genome)) < self.mutation_rate

        genome[mask] += (
            np.random.randn(np.sum(mask))
            * self.mutation_strength
        )

        return genome

    def create_next_generation(self, results):

        new_population = []

        # Preserve the best agents
        for i in range(self.elite_count):

            genome = results[i][1].network.get_genome().copy()

            elite = Agent(network_type=self.network_type)
            elite.network.set_genome(genome)

            new_population.append(elite)

        # Fill the remaining population with offspring
        while len(new_population) < self.population_size:

            parent1 = self.select_parent(results)
            parent2 = self.select_parent(results)

            child_genome = self.crossover(
                parent1,
                parent2
            )

            child_genome = self.mutate(child_genome)

            child = Agent(network_type=self.network_type)
            child.network.set_genome(child_genome)

            new_population.append(child)

        self.population = new_population

    def run(
            self,
            generations=500,
            pause_requested=None,
            resume_event=None,
            status_queue=None
    ):

        best_validation_average = float("inf")
        best_genome = None
        completed_generations = 0

        validation_interval = 5
        validation_games = 1000
        validation_seed_offset = 100000

        print("\nEvolution starting...")
        print("Method: CPU multiprocessing")
        print(f"Workers: {self.workers}")

        try:
            with ProcessPoolExecutor(
                    max_workers=self.workers
            ) as executor:
                baseline_average = validate_agent(
                    self.baseline_agent,
                    games=1000,
                    seed_offset=100000
                )

                print(
                    f"\nBaseline validation average: "
                    f"{baseline_average:.3f}"
                )
                for generation in range(generations):

                    print(
                        f"\n========== GENERATION {generation + 1}/{generations} ==========",
                        flush=True
                    )

                    # Evaluate the entire current population
                    training_seed_offset = generation * self.games_per_agent

                    results = self.evaluate_population(
                        executor,
                        seed_offset=training_seed_offset
                    )
                    diversity = self.calculate_diversity()

                    print(f"Population diversity: {diversity:.5f}")

                    current_best = results[0][0]

                    # Validate periodically using the same fixed games
                    should_validate = (
                            (generation + 1) % validation_interval == 0
                            or generation == 0
                            or generation == generations - 1
                    )

                    validation_average = None

                    if should_validate:

                        validation_average = validate_agent(
                            results[0][1],
                            games=validation_games,
                            seed_offset=validation_seed_offset
                        )

                        if validation_average < best_validation_average:

                            best_validation_average = validation_average

                            best_genome = (
                                results[0][1].network.get_genome().copy()
                            )

                            self.stagnation_counter = 0

                            self.save_checkpoint(
                                best_genome,
                                best_validation_average
                            )
                        else:
                            self.stagnation_counter += 1

                    # Adjust mutation based on stagnation
                    # Gradually increase mutation during stagnation
                    if self.stagnation_counter < 10:

                        mutation_factor = 1.0

                    elif self.stagnation_counter < 20:

                        mutation_factor = 1.5

                    elif self.stagnation_counter < 30:

                        mutation_factor = 2.0

                    else:

                        mutation_factor = 3.0

                    self.mutation_rate = min(
                        self.normal_mutation_rate * mutation_factor,
                        0.15
                    )

                    self.mutation_strength = min(
                        self.normal_mutation_strength * mutation_factor,
                        0.3
                    )

                    completed_generations += 1

                    print(
                        f"Generation {generation + 1}: "
                        f"Training best = {current_best:.3f} | "
                        f"Validation best = {best_validation_average:.3f} | "
                        f"Stagnation = {self.stagnation_counter} | "
                        f"Mutation = {self.mutation_strength:.3f}"
                    )

                    if validation_average is not None:
                        print(f"Current validation score: {validation_average:.3f}")

                    # Send progress to the GUI, if one is connected
                    if status_queue is not None:
                        status_queue.put((
                            "generation",
                            generation + 1,
                            current_best,
                            best_validation_average,
                            self.stagnation_counter,
                            self.mutation_strength,
                            best_genome.copy()
                        ))

                    # Pause only after a complete generation
                    if (
                            pause_requested is not None
                            and pause_requested.is_set()
                            and generation < generations - 1
                    ):

                        # Clear this before notifying the GUI.
                        # This prevents a quick Resume click from being lost.
                        resume_event.clear()

                        print("\nEvolution paused.")

                        if status_queue is not None:
                            status_queue.put((
                                "paused",
                                generation + 1,
                                best_validation_average,
                                best_genome.copy()
                            ))

                        # Wait here until the GUI signals Resume
                        resume_event.wait()

                        pause_requested.clear()

                        print("Evolution resumed.\n")

                        if status_queue is not None:
                            status_queue.put(("resumed",))

                    # Only create the next generation after resuming
                    if generation < generations - 1:
                        self.create_next_generation(results)

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user!")

        if best_genome is None:
            print("No complete generation was evaluated.")
            return None, None

        best_agent = Agent(network_type=self.network_type)
        best_agent.network.set_genome(best_genome)

        print("\nTraining finished.")
        print(f"Generations completed: {completed_generations}")
        print(f"Best validation average: {best_validation_average:.3f}")

        return best_agent, best_validation_average
