import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def compute(model, texts, tresh_hold=0.7, roc_curve=True):
    count_correct = 0
    count_all = 0
    for class_ in texts:
        for text in texts[class_]:
            p = model.predict(text)
            p = sorted(p.items(), key=lambda x: x[1], reverse=True)
            if p[0][0] == class_:
                count_correct += 1
            count_all += 1

    accuracy = count_correct / count_all

    print(f'Accuracy: {accuracy}')
