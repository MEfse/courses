from datetime import datetime


#intervals = [("09:00", "10:00"), ("09:30", "11:00"), ("11:00", "12:00")]
def find_schedule_conflicts(intervals):
    """
    Находит все конфликтующие пары интервалов времени, включая интервалы через полночь.
    
    Аргументы:
        intervals: список кортежей вида [("HH:MM", "HH:MM"), ...]
        
    Возвращает:
        Список кортежей конфликтующих пар интервалов [(interval1, interval2), ...]
    """
    conflicts = []
    intervals = [(datetime.strptime(start, "%H:%M"), datetime.strptime(end, "%H:%M")) for start, end in intervals]
    for i in range(len(intervals)):
        for j in range(i+1, len(intervals)):
            start1, end1 = intervals[i]
            start2, end2 = intervals[j]

        if (start1 < end2) and (start2 < end1):
            conflicts.append(((intervals[i][0].strftime("%H:%M"), intervals[i][1].strftime("%H:%M")),
                                   (intervals[j][0].strftime("%H:%M"), intervals[j][1].strftime("%H:%M"))))
    

    return conflicts

#find_schedule_conflicts(intervals)