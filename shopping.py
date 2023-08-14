import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    def month_to_number(month):
        """ Change month to integer """
        match month:
            case "Jan":
                return 0
            case "Feb":
                return 1
            case "Mar":
                return 2
            case "Apr":
                return 3
            case "May":
                return 4
            case "June":
                return 5
            case "Jul":
                return 6
            case "Aug":
                return 7
            case "Sep":
                return 8
            case "Oct":
                return 9
            case "Nov":
                return 10
            case "Dec":
                return 11


    evidence = []
    labels = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            evidence.append([
                int(row[0]),                                        # Administrative, an integer
                float(row[1]),                                      # Administrative_Duration, a floating point number
                int(row[2]),                                        # Informational, an integer
                float(row[3]),                                      # Informational_Duration, a floating point number
                int(row[4]),                                        # ProductRelated, an integer
                float(row[5]),                                      # ProductRelated_Duration, a floating point number
                float(row[6]),                                      # BounceRates, a floating point number
                float(row[7]),                                      # ExitRates, a floating point number
                float(row[8]),                                      # PageValues, a floating point number
                float(row[9]),                                      # SpecialDay, a floating point number
                month_to_number(row[10]),                           # Month, an index from 0 (January) to 11 (December)
                int(row[11]),                                       # OperatingSystems, an integer
                int(row[12]),                                       # Browser, an integer
                int(row[13]),                                       # Region, an integer
                int(row[14]),                                       # TrafficType, an integer
                int(1 if row[15] == "Returning_Visitor" else 0),    # Returning_Visitor
                int(1 if row[16] == "TRUE" else 0)                  # Weekend, an integer 0 (if false) or 1 (if true)
                ])
            
            labels.append(
                int(1 if row[17] == "TRUE" else 0)                  # Revenue, 0 if False, 1 if True
            )
            
        if len(evidence) != len(labels):
            sys.exit("Error while loading data")

        return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    
    # Variables for amont of true/false in labels and correct predictions
    label_true = 0
    pred_true = 0
    label_false = 0
    pred_false = 0

    # Loop through labels, and predictions
    for label, pred in zip(labels, predictions):
        if label == 1:
            label_true += 1
            if pred == 1:
                pred_true += 1
        if label == 0:
            label_false += 1
            if pred == 0:
                pred_false += 1

    # Compute results
    sensitivity = pred_true / label_true
    specifity = pred_false / label_false
            
    return (sensitivity, specifity)


if __name__ == "__main__":
    main()
