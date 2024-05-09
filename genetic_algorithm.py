import random
from copy import deepcopy

from utils import max_component_size


def create_fitness(words, grid_size):
  # returns fitness function
  
  words_dict = {w: i for i, w in enumerate(words)}
  
  def fitness(individual):
    res = 0
    
    horizontal = {i: [] for i in range(grid_size)}
    vertical = {i: [] for i in range(grid_size)}

    # Word is out of grid
    # Penalize by each letter out of grid
    for entry in individual:
      x, y, direction, word = entry 
      if direction == 0:
        horizontal[x].append([y, word])
        if y + len(word) - 1 >= grid_size:
          res += y + len(word) - grid_size
      else:
        vertical[y].append([x, word])
        if x + len(word) - 1 >= grid_size:
          res += x + len(word) - grid_size

    # Parallel neighboring words
    # Penalize for each letter neighboring
    for i in range(grid_size - 1):
      for y, word in horizontal[i]:
        ymin = y
        ymax = y + len(word) - 1
        for yi, wordi in horizontal[i + 1]:
          if yi <= ymax and yi + len(wordi) >= ymin:
            dif =  max(0, min(ymax, yi + len(wordi) - 1) - max(ymin, yi) + 1)
            
            # Fine if vertical word crosses them and the beginning of one and the end of another word
            crossed_by_vertical = False
            if dif == 1:
              if yi == ymax:
                 for (x, wordk) in vertical[yi]:
                    if x <= i < x + len(wordk) - 1:
                       crossed_by_vertical = True
                       break
              if ymin == yi + len(wordi) - 1:
                 for (x, wordk) in vertical[ymin]:
                    if x <= i < x + len(wordk) - 1:
                       crossed_by_vertical = True
                       break     
            if not crossed_by_vertical:
               res += dif

      
      for x, word in vertical[i]:
        xmin = x
        xmax = x + len(word) - 1
        for xi, wordi in vertical[i + 1]:
          if xi <= xmax and xi + len(wordi) >= xmin:
            dif = max(0, min(xmax, xi + len(wordi) - 1) - max(xmin, xi) + 1) 
            # Fine if horizontal word crosses them and the beginning of one and the end of another word
            crossed_by_horizontal = False
            if dif == 1:
              if xi == xmax:
                 for (y, wordk) in horizontal[xi]:
                    if y <= i < y + len(wordk) - 1:
                       crossed_by_horizontal = True
                       break
              if xmin == xi + len(wordi) - 1:
                 for (y, wordk) in horizontal[xmin]:
                    if y <= i < y + len(wordk) - 1:
                       crossed_by_horizontal = True
                       break     
            if not crossed_by_horizontal:
               res += dif
      
    
    # Overlay
    # Penalize for each overlayed letter
    for i in range(grid_size):
      for j, (y, word) in enumerate(horizontal[i]):
        ymin = y - 1
        ymax = y + len(word) - 1
        for yi, wordi in horizontal[i][j + 1:]:
          if yi - 1 <= ymax and yi + len(wordi) - 1 >= ymin:
            res += max(0, min(ymax, yi + len(wordi) - 1) - max(ymin, yi - 1) + 1)

      
      for j, (x, word) in enumerate(vertical[i]):
        xmin = x - 1 
        xmax = x + len(word) - 1
        for xi, wordi in vertical[i][j + 1:]:
          if xi - 1 <= xmax and xi + len(wordi) - 1 >= xmin:
            
            res += max(0, min(xmax, xi + len(wordi) - 1) - max(xmin, xi - 1) + 1) 

    # Incorrect crosses
    # Count wrong crosses
    edges = []
    wrong_crosses = 0

    for i in range(grid_size):
      for y, wordy in horizontal[i]:
        for j in range(max(y - 1, 0), min(y + len(wordy) + 1, grid_size)):
          for x, wordx in vertical[j]:
            if x <= i < x + len(wordx):
              if y <= j < y + len(wordy):
                edges.append([words_dict[wordx], words_dict[wordy]]) 
                if wordx[i - x] != wordy[j - y]:
                  wrong_crosses += 1
                  
              elif (y + len(wordy) == j or y == j + 1):
                res += 1
            if (x + len(wordx) == i or x == i + 1) and 0 <= j - y < len(wordy):
              res += 1

    # Connectivity
    max_component = max_component_size(len(words), edges)

    # Scale wrong_crosses by size of max component
    # So wrong crosses, when crossword is already built, penalizes more
    res += wrong_crosses * max_component

    # Penalize by the number of words left to connect to the greates component
    res += len(words) - max_component

    return -res

  return fitness

def get_individual(words, grid_size):
  # Individual is represented as list:
  # [x_coord, y_coord, direction (vertical/horizontal), word]
  individual = []
  
  for i in range(len(words)):
    word = words[i]
    # direction: 0 = vertical, 1 = horizontal
    direction = random.choice([0, 1])
    # Initally generate words to be inside grid
    if direction == 0:
      x = random.randint(0, grid_size - 1)
      y = random.randint(0, grid_size - 1 - len(word))
    else :
      x = random.randint(0, grid_size - 1 - len(word))
      y = random.randint(0, grid_size - 1)
    individual.append([x, y, direction, word])
  return individual

def initial_population(words, population_size, fitness, grid_size):
  population = [get_individual(words, grid_size) for _ in range(population_size)]
  population.sort(key=lambda x: fitness(x))
  return population

def get_parents(population, n_offsprings, fitness):
  # Take the most fit parents using tournament

  parents = []
  for _ in range(2 * n_offsprings):
    best_fitness = -1e9
    best = None
    for j in range(10):
      i = random.choice(range(len(population)))
      cur_fitness = fitness(population[i])
      if best_fitness < cur_fitness:
        best = population[i]
        best_fitness = cur_fitness
    parents.append(deepcopy(best))
  mothers = parents[::2]
  fathers = parents[1::2]
  return mothers, fathers

def cross(mother, father, prob=0.9):
  # cross two parents together
  # with probability prob take father's x_coord and direction
  offspring = []
  if random.choices([0, 1], weights=[1 - prob, prob])[0]:
    for m, f in zip(mother, father):
      offspring.append([f[0], m[1], f[2], m[3]])
  else:
    offspring = deepcopy(mother)

  return deepcopy(offspring)

def mutate(offspring, prob=0.9, grid_size=20):
  # mutate offspring 
  
  # with probability prob randomly change individual word
  if random.choices([0, 1], weights=[1 - prob, prob])[0]:
    # choose random word
    i = random.choice(range(len(offspring)))
    direction = offspring[i][2]

    # with probability prob change direction
    if random.choice([0, 1]):
      direction = 1 - direction
    # randomly move inside grid
    if direction == 0:
      x = random.randint(0, grid_size - 1)
      y = random.randint(0, grid_size - 1 - len(offspring[i][3]))
    else :
      x = random.randint(0, grid_size - 1 - len(offspring[i][3]))
      y = random.randint(0, grid_size - 1)
    offspring[i][0] = x
    offspring[i][1] = y
    offspring[i][2] = direction
  
  # with probabilty 1 - prob swap positions and directions of wordsand add random shift to them
  if random.choices([0, 1], weights=[prob, 1 - prob])[0]:
    i, j = random.choices(range(len(offspring)), k=2)
    
    # swap words
    offspring[i][0], offspring[j][0] = offspring[j][0], offspring[i][0]
    offspring[i][1], offspring[j][1] = offspring[j][1], offspring[i][1]
    offspring[i][2], offspring[j][2] = offspring[j][2], offspring[i][2]

    # shift words to leave them inside grid
    dx = dy = 0
    if offspring[i][2] == 0:
      dx = random.randint(-len(offspring[i][3]) + 1, len(offspring[i][3]) - 1)
    else:
      dy = random.randint(-len(offspring[i][3]) + 1, len(offspring[i][3]) - 1)
    offspring[i][0] += dx
    offspring[i][1] += dy
    offspring[i][0] = offspring[i][0] if offspring[i][0] >= 0 else 0
    offspring[i][0] = offspring[i][0] if offspring[i][0] < grid_size else grid_size - len(offspring[i][3])
    
    offspring[i][1] = offspring[i][1] if offspring[i][1] >= 0 else 0
    offspring[i][1] = offspring[i][1] if offspring[i][1] < grid_size else grid_size - len(offspring[i][3])
    
    dx = dy = 0
    if offspring[j][2] == 0:
      dx = random.randint(-len(offspring[j][3]) + 1, len(offspring[j][3]) - 1)
    else:
      dy = random.randint(-len(offspring[j][3]) + 1, len(offspring[j][3]) - 1)
    offspring[j][0] += dx
    offspring[j][1] += dy
    offspring[j][0] = offspring[j][0] if offspring[j][0] >= 0 else 0
    offspring[j][0] = offspring[j][0] if offspring[j][0] < grid_size else grid_size - len(offspring[i][3])
    offspring[j][1] = offspring[j][1] if offspring[j][1] >= 0 else 0
    offspring[j][1] = offspring[j][1] if offspring[j][1] < grid_size else grid_size - len(offspring[i][3])
    
  return offspring

def replace_population(population, new_individuals, fitness, save_part=0.1):
    # leave best (save_part * 100)% of previous population, add new ones and leave the best ones
    size = len(population)
    population.sort(key=lambda x: fitness(x))
    population=population[-int(save_part * size):]
    population.extend(new_individuals)
    population.sort(key=lambda x: fitness(x))
    return population[-size:]

def evolution_step(population, fitness, n_offsprings, grid_size):
  mothers, fathers = get_parents(population, n_offsprings, fitness)
  offsprings = []
  for mother, father in zip(mothers, fathers):
      offspring = mutate(cross(mother, father), grid_size=grid_size)
      offsprings.append(offspring)
  
  new_population = replace_population(population, offsprings, fitness)
  return new_population

def evolution(words, population_size=300, n_offsprings=300, generations=10000, restart_interval=1000, grid_size=20, verbose=False):
  fitness = create_fitness(words, grid_size)
  population = initial_population(words, population_size, fitness, grid_size)

  for generation in range(generations):
    population = evolution_step(population, fitness, n_offsprings, grid_size)
    best_individual = population[-1]
    best_fitness = fitness(best_individual)
    # restart each restart_interval generations 
    # probably does not converge for early mistakes (choice where to start building)
    if (generation + 1) % restart_interval == 0:
      population = initial_population(words, population_size, fitness, grid_size)
    
    if verbose:
      print(f'Generation {generation + 1}: {best_fitness}')
    # stop when crossword is built
    if best_fitness == 0:
      break

  return best_individual, best_fitness