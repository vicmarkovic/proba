import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def compute(model, texts, tresh_hold=0.7, roc_curve=True):
    '''
    we assume there is only two classes
    tresh_hold: depending from first class we will find threshold maybe needs to first be treshold or 1 - treshold
    '''

    True_positive = 0
    False_negative = 0
    True_negative = 0
    False_positive = 0
    y = []
    positive, negative = texts.keys()
    for text in texts[positive]:
        p = model.predict(text)[1]
        y.append(p[positive])
        if p[positive] > tresh_hold:
            True_positive += 1
        else:
            False_negative += 1

    for text in texts[negative]:
        p = model.predict(text)[1]
        y.append(p[positive])
        if p[positive] <= tresh_hold:
            True_negative += 1
        else:
            False_positive += 1

    accuracy = (True_positive + True_negative) / (True_positive + False_negative + True_negative + False_positive)
    sensitivity = True_positive / (True_positive + False_negative)
    specificity = True_negative / (True_negative + False_positive)
    precision = True_positive / (True_positive + False_positive)
    f1_score = 2 * (sensitivity * precision) / (sensitivity + precision)
    print(f'Accuracy: {accuracy}')
    print(f'if {positive} is positive')
    print(f'sensitivity = {sensitivity}')
    print(f'specificity = {specificity}')
    print(f'precision = {precision}')
    print(f'f1_score = {f1_score}')

    if roc_curve:
        y_true = [1]*len(texts[positive]) + [0]*len(texts[negative])
        get_roc(y_true, y)


    return {'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score}

def get_roc(y_true, y):
    fpr, tpr, thresholds = roc_curve(y_true, y)

    # Calculate AUC-ROC (Area Under the ROC Curve)
    auc_roc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.2f}')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve with Sensitivity and Specificity')
    plt.legend()
    plt.show()

def histogram(model, texts):
    probs = {}
    for class_ in texts:
        probs[class_] = []
        for text in texts[class_]:
            p = model.predict(text)[1]
            probs[class_].append(p['ham'])
    color = 'red'
    n = 1
    for class_ in probs:
        plt.scatter(np.arange(len(probs[class_])), probs[class_], color=color)
        color = 'blue'
        n = 2
    plt.show()
