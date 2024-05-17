import numpy as np

def target_function(x):
    return 186 * x **3 - 7.22 * x ** 2 + 15.5 * x - 13.2

def binary_to_decimal(num):
    result = 0
    for i in range(len(num)):
        result += 2 ** (3 - i - 1) * num[i]
    return result

population_size = 50
generations = 100
mutation_rate = 0.1
binary_length = 20 
report_step = 20


population = np.random.choice([0, 1], size=(population_size, binary_length), replace=True)

for generation in range(generations+1):
    decimal_values = np.apply_along_axis(binary_to_decimal, 1, population)

    fitness_values = np.abs(target_function(decimal_values))

    selected_indices = np.argsort(fitness_values)[:int(population_size / 2)]

    crossover_indices = np.random.choice(selected_indices, size=int(population_size / 2), replace=True)
    offspring = np.vstack([population[selected_indices], population[crossover_indices]])

    mutation_mask = np.random.rand(population_size, binary_length) < mutation_rate
    offspring ^= mutation_mask

    population = offspring

    if generation % report_step == 0:
        best_solution_binary = population[np.argmin(np.abs(target_function(np.apply_along_axis(binary_to_decimal, 1, population))))]
        gen_best_solution = binary_to_decimal(best_solution_binary)
        gev_best_value = target_function(gen_best_solution)
        print(f"=========== Gen {generation} ===========")
        print(f"Best solution:\t {gen_best_solution}")
        print(f"Function value:\t {gev_best_value}")