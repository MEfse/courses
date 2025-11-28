import math

X = [
    [1, 2, 3],
    [4, 5, 6],
    [1, 2, 3],
    [7, 8, 9]
]

X = [
        [0, 0],
        [3, 4]
    ]

def find_twin_pairs(X, threshold=5.0):
    """
    Находит все пары объектов, у которых евклидово расстояние меньше threshold.
    
    Аргументы:
    X -- двумерный список чисел (n x m)
    threshold -- пороговое значение расстояния
    
    Возвращает:
    Список кортежей (i, j, distance), где i < j и distance < threshold
    """
    pairs = []
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            result = 0
            for k in range(len(X[i])):
                result += (X[i][k] - X[j][k])**2
    
            distance = math.sqrt(result)
            
            if distance < threshold:
                pairs.append((i, j, distance))
               
    return pairs


find_twin_pairs(X)