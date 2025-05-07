import numpy as np

def flatten_coefficients(a, b1, b2, c):
    """
    Flattens the RK coefficients into a single list.
    
    Parameters:
    - a, b1, b2: Lists or numpy arrays of RK coefficients.
    - c: A list of lists or a numpy object array of RK coefficients.
    
    Returns:
    - A single flat list containing all coefficients.
    """
    flat_list = []
    # Flatten 'a', 'b1', 'b2'
    for coeff in (a, b1, b2):
        flat_list.extend(coeff)
    # Flatten 'c', handling potential nested structures
    for sublist in c:
        flat_list.extend(sublist)
    return flat_list

def reconstruct_coefficients(flat_list, stage):
    """
    Reconstructs the RK coefficients from a flat list for a given stage.
    
    Parameters:
    - flat_list: The flat list containing all RK coefficients.
    - stage: The number of stages for the RK method.
    
    Returns:
    - The reconstructed a, b1, b2, and c coefficients.
    """
    # Calculate lengths of each coefficient array
    a_len = stage
    b_len = stage
    c_lengths = [len(c_row) for c_row in generate_c_structure(stage)]
    
    # Extract a, b1, b2, and c from the flat list
    a = flat_list[:a_len]
    b1 = flat_list[a_len:a_len + b_len]
    b2 = flat_list[a_len + b_len:a_len + 2 * b_len]
    c_start = a_len + 2 * b_len
    c = []
    for length in c_lengths:
        c.append(flat_list[c_start:c_start + length])
        c_start += length
    
    return a, b1, b2, c

def generate_c_structure(stage):
    """
    Generates the c structure based on the stage.
    
    Parameters:
    - stage: The number of stages for the RK method.
    
    Returns:
    - A nested list structure representing 'c' coefficients.
    """
    return [[]] + [[1 / (i + 1) for _ in range(i)] for i in range(1, stage)]

def main():
    # Example RK method coefficients
    stage = 6
    a = [0, 1/5, 3/10, 3/5, 1, 7/8]
    b1 = [37/378, 0, 250/621, 125/594, 0, 512/1771]
    b2 = [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4]
    c = [
        [],
        [1/5],
        [3/40, 9/40],
        [3/10, -9/10, 6/5],
        [-11/54, 5/2, -70/27, 35/27],
        [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096]
    ]

    # Flatten coefficients
    flat_list = flatten_coefficients(a, b1, b2, c)
    print("Flat List:", flat_list)

    # Reconstruct coefficients
    a_reconstructed, b1_reconstructed, b2_reconstructed, c_reconstructed = reconstruct_coefficients(flat_list, stage)
    print("\nReconstructed Coefficients:")
    print("a:", a_reconstructed)
    print("b1:", b1_reconstructed)
    print("b2:", b2_reconstructed)
    print("c:", c_reconstructed)
def main2():
    # Example RK method coefficients
    stage = 2
    a = [0, 1]
    b1 = [0.5,0.5]
    b2 = [1,0]
    c = [
        [],
        [1],

    ]

    # Flatten coefficients
    flat_list = flatten_coefficients(a, b1, b2, c)
    print("Flat List:", flat_list)

    # Reconstruct coefficients
    a_reconstructed, b1_reconstructed, b2_reconstructed, c_reconstructed = reconstruct_coefficients(flat_list, stage)
    print("\nReconstructed Coefficients:")
    print("a:", a_reconstructed)
    print("b1:", b1_reconstructed)
    print("b2:", b2_reconstructed)
    print("c:", c_reconstructed)
if __name__ == '__main__':
    main2()
