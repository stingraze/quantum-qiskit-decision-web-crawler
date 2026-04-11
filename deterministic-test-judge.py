import random

class DeterministicTestJudge:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def generate_synthetic_data(self, num_samples):
        # Generate synthetic data for the tests
        return [random.randint(1, 100) for _ in range(num_samples)]

    def evaluate(self, input_data):
        # Algorithm-specific evaluation logic
        if self.algorithm == 'algorithm_a':
            return self.algorithm_a_behavior(input_data)
        elif self.algorithm == 'algorithm_b':
            return self.algorithm_b_behavior(input_data)
        else:
            raise ValueError("Undefined Algorithm")

    def algorithm_a_behavior(self, data):
        # Mock implementation for algorithm A
        return [x * 2 for x in data]

    def algorithm_b_behavior(self, data):
        # Mock implementation for algorithm B
        return [x + 10 for x in data]

    def score(self, results):
        # Compute a deterministic score based on the results
        return sum(results) / len(results)

if __name__ == '__main__':
    test_judge = DeterministicTestJudge('algorithm_a')
    synthetic_data = test_judge.generate_synthetic_data(10)
    results = test_judge.evaluate(synthetic_data)
    score = test_judge.score(results)
    print(f"Synthetic Data: {synthetic_data}")
    print(f"Results: {results}")
    print(f"Score: {score}")