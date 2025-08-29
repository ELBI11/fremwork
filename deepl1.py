import numpy as np

def square(z):
  return np.square(z)

def squarederivative(z):
  return 2 * z

 
def sigmoid(z):
 return 1 / (1 + np.exp(-z))
 
def sigmoid_derivative(z):
 s = sigmoid(z)
 return s * (1- s)

def divide(a, b):
 assert b != 0, "The divisor cannot be zero."
 return a / b
 
'''print(divide(10, 2)) # OK : 5.0
print(divide(10, 0)) # AssertionError: The divisor cannot be null'''

def calculer_carre(n):
 assert isinstance(n, int), "The input must be an integer."
 assert n >= 0, "The input must be non-negative."
 return n * n
 
print(calculer_carre(4)) # OK : 16
 # print(calculer_carre(-1)) # AssertionError: The input must be non-negative


def sommepositifs(lst):
 assert all(x > 0 for x in lst), "Tous les lments doivent tre positifs"
 total = 0
 for x in lst:
  total += x
 assert total > 0, "La somme doit rester positive"
 return total
 
print(sommepositifs([1, 2, 3])) # OK : 6
# print(sommepositifs([1,-2, 3])) # AssertionError: "All elements must be positive"



'''def nom_de_la_fonction(param: np.ndarray)-> np.ndarray:
 """Description de la fonction.
 4 Parameters:
 5
 param (np.ndarray): Tableau NumPy d’entree.
 6 Returns:
 7
 np.ndarray: Tableau NumPy resultant.
 8 """
 # Operations sur param (vectorisees)
 assert # assert sur les prams
 assert # assert de validation
 resultat = ... # Calculs utilisant des fonctions NumPy
 return resultat'''

def square_array(arr: np.ndarray)-> np.ndarray:
 """ Compute the square of each element in the input array.
 4 Parameters:
 5 arr (np.ndarray): Input array of any shape.
 6 Returns:
 7 np.ndarray: Array with squared elements, same shape as
 input. """
 # Validate input
 assert isinstance(arr, np.ndarray), "Input must be a NumPy ndarray"
 assert arr.size > 0, "Input array cannot be empty"
 result = np.square(arr)
 # Validate output
 assert isinstance(result, np.ndarray), "Output must be a NumPy ndarray"
 assert result.shape == arr.shape, "Output shape must match input shape"
 return result
square_array(np.array([[1, 2], [3, 4]]))


import numpy as np
def relu_array(arr: np.ndarray)-> np.ndarray:
 """Apply the ReLU activation function to each element in the
 input array.
 4 Parameters:
 5 arr (np.ndarray): Input array of any shape.
 6 Returns:
 7 np.ndarray: Array with ReLU applied, same shape as input.
 """
 # Validate input
 assert isinstance(arr, np.ndarray), "Input must be a NumPy ndarray"
 assert arr.size > 0, "Input array cannot be empty"
 result = np.maximum(0, arr)
 # Validate output
 assert isinstance(result, np.ndarray), "Output must be a NumPy ndarray"
 assert result.shape == arr.shape, "Output shape must matche input shape"
 assert np.all(result >= 0), "ReLU output must be non-negative"
 return result
relu_array(np.array([-1, 0, 1]))


def normalize_array(arr: np.ndarray)-> np.ndarray:
 """ Normalize the input array to have values between 0 and 1.
 4 Parameters:arr (np.ndarray): Input array of any shape.
 5 Returns:np.ndarray: Normalized array, same shape as input."""
 # Validate input
 assert isinstance(arr, np.ndarray), "Input must be a NumPy ndarray"
 assert arr.size > 0, "Input array cannot be empty"
 arr_min = np.min(arr)
 arr_max = np.max(arr)
 # Validate range
 assert arr_max != arr_min, "Input array cannot be constant"
 result = (arr- arr_min) / (arr_max- arr_min)



  # Validate output
 assert isinstance(result, np.ndarray), "Output must be a NumPy ndarray"
 assert result.shape == arr.shape, "Output shape must match input shape"
 assert np.all((result >= 0) & (result <= 1)), "Output must be between 0 and 1"
 return result
print(normalize_array(np.array([1, 2, 3, 4])))


def relu_array(arr: np.ndarray) -> np.ndarray:
    # Validate input
    assert isinstance(arr, np.ndarray), "Input must be a NumPy ndarray"
    assert arr.size > 0, "Input array cannot be empty"
    
    # Apply ReLU
    result = np.maximum(0, arr)
    
    # Validate output
    assert isinstance(result, np.ndarray), "Output must be a NumPy ndarray"
    assert result.shape == arr.shape, "Output shape must match input shape"
    assert np.all(result >= 0), "ReLU output must be non-negative"
    
    return result

def ta_array(arr: np.ndarray) -> np.ndarray:
    # Validate input
    assert isinstance(arr, np.ndarray), "Input must be a NumPy ndarray"
    assert arr.size > 0, "Input array cannot be empty"
    
    # Compute tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    exp_pos = np.exp(arr)
    exp_neg = np.exp(-arr)
    result = (exp_pos - exp_neg) / (exp_pos + exp_neg)
    
    # Validate output
    assert isinstance(result, np.ndarray), "Output must be a NumPy ndarray"
    assert result.shape == arr.shape, "Output shape must match input shape"
    assert np.all((result >= -1) & (result <= 1)), "Tanh output must be between -1 and 1"
    
    return result

