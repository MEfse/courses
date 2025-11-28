import re

def sum_numbers_in_string(input_string: str) -> int:
    """
    Находит все целые числа в строке и возвращает их сумму.
    
    Args:
        input_string: Строка, в которой нужно найти числа.
        
    Returns:
        Сумма всех найденных целых чисел.
    """
    total_sum = 0
    current_number = 0
    for char in input_string:
        if char.isdigit(): 
            current_number = current_number * 10 + int(char)
        else:
            total_sum += current_number
            current_number = 0

    total_sum += current_number        
    return total_sum

def sum_numbers_in_string_regex(input_string: str) -> int:
    """
    Находит все целые числа в строке и возвращает их сумму.
    
    Args:
        input_string: Строка, в которой нужно найти числа.
        
    Returns:
        Сумма всех найденных целых чисел.
    """
    


#sum_numbers_in_string(s)