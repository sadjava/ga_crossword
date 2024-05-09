from matplotlib import pyplot as plt

def max_component_size(num_vertices, edges):
    # Create an adjacency list to represent the graph
    adj_list = {i: [] for i in range(num_vertices)}
    for edge in edges:
        u, v = edge
        adj_list[u].append(v)
        adj_list[v].append(u)

    # Initialize a set to keep track of visited vertices
    visited = set()

    # Initialize the component count
    components = 0

    # Perform DFS
    def dfs(node):
        visited.add(node)
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                dfs(neighbor)

    # Iterate through all vertices
    component_sizes = []
    visited_sizes = [0]
    for vertex in range(num_vertices):
        if vertex not in visited:
            component_sizes.append(len(visited) - visited_sizes[-1])
            visited_sizes.append(len(visited))
            components += 1
            dfs(vertex)
    
    component_sizes.append(len(visited) - visited_sizes[-1])
    return max(component_sizes) 

def plot_crossword(crossword, fitness_score, grid_size=20):
  # plots cities and route

  plt.figure(figsize=(8, 8))
  ax = plt.subplot(111)

  plt.title(f"Fitness: {fitness_score}")
  # Draw grid lines
  for i in range(grid_size + 1):
      ax.plot([i, i], [0, grid_size], color='black', linewidth=2)
      ax.plot([0, grid_size], [i, i], color='black', linewidth=2)

  # Draw each word letter by letter
  for entry in crossword:
      y, x, direction, text = entry
      for i, letter in enumerate(text):
          if direction == 0:  # Vertical (down)
              plt.text(x + i + 0.5, grid_size - (y + 0.5), letter, ha='center', va='center', fontsize=10)
          else:  # Horizontal (right)
              plt.text(x + 0.5, grid_size - (y + i + 0.5), letter, ha='center', va='center', fontsize=10)

  # Set axis properties
  ax.set_xlim([0, grid_size + 1])
  ax.set_ylim([0, grid_size + 1])
  ax.set_aspect('equal', 'box')
  ax.axis('off')
  plt.show()