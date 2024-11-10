import numpy as np

def flatten_coefficients(a, b1, b2, c):
    """
    Flattens the RK45 coefficients into a single list.
    
    Parameters:
    - a, b1, b2: Lists or numpy arrays of RK45 coefficients.
    - c: A list of lists or a numpy object array of RK45 coefficients.
    
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

def reconstruct_coefficients(flat_list):
    """
    Reconstructs the RK45 coefficients from a flat list.
    
    Parameters:
    - flat_list: The flat list containing all RK45 coefficients.
    
    Returns:
    - The reconstructed a, b1, b2, and c coefficients.
    """
    # Indices in the flat list where each coefficient starts
    a_start, a_end = 0, 6
    b1_start, b1_end = 6, 12
    b2_start, b2_end = 12, 18
    c_start = 18
    
    a = flat_list[a_start:a_end]
    b1 = flat_list[b1_start:b1_end]
    b2 = flat_list[b2_start:b2_end]
    c = [
         [],
         flat_list[c_start:c_start+1],
         flat_list[c_start+1:c_start+3],
         flat_list[c_start+3:c_start+6],
         flat_list[c_start+6:c_start+10],
         flat_list[c_start+10:c_start+15],
         ]
    
    return a, b1, b2, c
def main():
    # Example Runge–Kutta–Fehlberg method
    a = [0, 1/4, 3/8, 12/13, 1, 1/2]
    b1 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
    b2 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
    c = [
        [],
        [1/4],
        [3/32, 9/32],
        [1932/2197, -7200/2197, 7296/2197],
        [439/216, -8, 3680/513, -845/4104],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ]


    # Example Runge–Kutta–Fehlberg method
    a = [0, 1/5, 3/10, 3/5, 1, 7/8]
    b1 = [37/378, 0, 250/621, 125/594, 0, 512/1771]
    b2 = [2825/27648, 0, 18575/48384	, 13525/55296, 277/14336, 1/4]
    c = [
        [],
        [1/5],
        [3/40, 	9/40],
        [	3/10, 	-9/10, 6/5],
        [-11/54, 	5/2, -70/27, 	35/27],
        [1631/55296, 175/512, 575/13824,44275/110592	, 253/4096]
    ]

    # Flatten coefficients
    flat_list = flatten_coefficients(a, b1, b2, c)
    print("Flat List:", flat_list)

    # Reconstruct coefficients
    a_reconstructed, b1_reconstructed, b2_reconstructed, c_reconstructed = reconstruct_coefficients(flat_list)
    print("\nReconstructed Coefficients:")
    print("a:", a_reconstructed)
    print("b1:", b1_reconstructed)
    print("b2:", b2_reconstructed)
    print("c:", c_reconstructed)

if __name__ == '__main__':
    main()

