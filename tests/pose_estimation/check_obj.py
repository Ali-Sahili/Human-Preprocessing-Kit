
import trimesh
import numpy as np


def get_k_nearest_vertices(vertices, target_index = 6073, k=10):
  target_vertex = vertices[target_index]
  distances = np.linalg.norm(vertices - target_vertex, axis=1)
  nearest_indices = np.argsort(distances)[1:k+1]  # Skip the first index (itself)
  print("Indices of the 10 nearest vertices:", nearest_indices)
  return nearest_indices

# Function to calculate the distance from a point to the line segment
def distance_to_line_segment(point, vertex1, vertex2, line_vec, line_len_sq):
    point_vec = point - vertex1
    projection_factor = np.dot(point_vec, line_vec) / line_len_sq
    projection_factor = max(0, min(1, projection_factor))
    closest_point = vertex1 + projection_factor * line_vec
    return np.linalg.norm(point - closest_point)


# Load the mesh
mesh = trimesh.load('outputs/vibe/sample_video/meshes/0001/000000.obj')
vertices = mesh.vertices
faces = mesh.faces


mid_vertex = (vertices[2204]+vertices[2319])/2

distances = np.linalg.norm(vertices - mid_vertex, axis=1)
nearest_indices = np.argsort(distances)[1:11]  # Skip the first index (itself)
print("Indices of the 10 nearest vertices:", nearest_indices)
  

# get_k_nearest_vertices( vertices, 
#                         target_index = 2367, 
#                         k=5)

# vertex1 = vertices[4]
# vertex2 = vertices[6089]

# # Define the line segment vector and its length squared
# line_vec = vertex2 - vertex1
# line_len_sq = np.dot(line_vec, line_vec)
# distances = np.array([
#     distance_to_line_segment(v, 
#                             vertex1, 
#                             vertex2, 
#                             line_vec, 
#                             line_len_sq) 
#                     for v in vertices])
# num_nearest = 10
# nearest_indices = np.argsort(distances)[:num_nearest]
# print("Indices of the nearest vertices:", nearest_indices)