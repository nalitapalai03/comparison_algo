def matrix(predictions, test_data):
    # preparing confusion matrix
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    correct = 0
    length = len(test_data)
    for i in range(length):
        if test_data[i][-1] == 1:
            if test_data[i][-1] == predictions[i]:
                tp += 1
            else:
                fp += 1
        else:
            if test_data[i][-1] == predictions[i]:
                tn += 1
            else:
                fn += 1

        #print("{:20}{:20}".format(test_data[i], predictions[i]))

    # print the confusion matix
    print("\n\n\n\t\tCONFUSION MATRIX\n--------------------------------\n")
    print("{:12}{:10}\n{:12}{:10}\n".format(tp, fn, fp, tn))
    #print("POSITIVE : ", p)
    #print("NEGATIVE : ", n)
    TPR = tp / (tn + tp)
    TNR = tn / (tn + fp)
    PPV = tp / (tp + fp)
    FNR = 1 - TPR
    acc = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    print("PRECISION : {}".format(PPV * 100))
    print("MISS RATE : {}".format(FNR * 100))
    print("ACCURACY : {}% ".format(acc * 100))
    print("RECALL : {}% ".format(recall*100))

