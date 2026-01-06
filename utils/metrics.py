from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def classifier_evaluate(label, pred):
    acc = accuracy_score(label, pred)
    wp = precision_score(label, pred, average='weighted', zero_division=0)
    uar = recall_score(label, pred, average='macro', zero_division=0)
    waf = f1_score(label, pred, average='weighted', zero_division=0)
    cm = confusion_matrix(label, pred)

    return acc, wp, uar, waf, cm