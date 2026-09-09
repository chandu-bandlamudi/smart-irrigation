from numpy import dot
from numpy.linalg import norm

class Environment:
    def action(self, train_data, train_label, test_data):
        label = None
        score = 0
        limit = min(100, len(train_data))
        for i in range(limit):
            denominator = norm(train_data[i]) * norm(test_data)
            if denominator == 0:
                similarity = 0
            else:
                similarity = dot(train_data[i], test_data) / denominator
            if similarity > score:
                score = similarity
                label = train_label[i]
        return label
