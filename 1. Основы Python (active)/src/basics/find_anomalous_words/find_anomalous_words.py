def find_anomalous_words(text: str) -> list[str]:
    """
    Находит слова, длина которых отличается от средней длины слов в тексте более чем на 2 символа.

    :param text: Входная строка.
    :return: Список аномальных слов.
    """
    if text == "":
        return []

    len_words, count_words = 0, 0 
    margin=2

    words = text.strip().replace('!', '').replace('.', '').split()

    print(words)
    for word in words:
        len_words += len(word)
        count_words += 1

    avg_len = len_words / count_words
    anomaly_words = []

    for word in words:
        if (len(word) >= avg_len + margin) or (len(word) <= avg_len - margin):
            anomaly_words.append(word) 
            
    return anomaly_words