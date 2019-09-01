import matplotlib.pyplot as plt
import pandas as pd

def read_results(mtl_results_multiclass, mtl_results_multilabel, validation):
#read results files and return metrics results as lists ready to be printed

    multiclass_file = open(mtl_results_multiclass)
    multiclass_results = multiclass_file.readlines()

    multilabel_file = open(mtl_results_multilabel)
    multilabel_results = multilabel_file.readlines()

    epoch_mc = []
    accuracy_mc = []
    balanced_accuracy_mc = []
    loss_mc = []
    f1_mc = []
    precision_mc = []
    recall_mc = []

    if validation:
        for row in multiclass_results[1:]:
            #print(row.split(','))
            epoch_mc.append(int(row.split(',')[0]))
            accuracy_mc.append(float(row.split(',')[1]))
            balanced_accuracy_mc.append(float(row.split(',')[2]))
            #f1_mc.append(float(row.split(',')[3]))
            #precision_mc.append(float(row.split(',')[4]))
            #recall_mc.append(float(row.split(',')[5]))
            loss_mc.append(float(row.split(',')[3]))
            f1_mc.append(float(row.split(',')[4]))
            precision_mc.append(float(row.split(',')[5]))
            recall_mc.append(float(row.split(',')[6]))
    else:
        for row in multiclass_results:
            #print(row.split(','))
            epoch_mc.append(int(row.split(',')[0]))
            accuracy_mc.append(float(row.split(',')[1]))
            balanced_accuracy_mc.append(float(row.split(',')[2]))
            #f1_mc.append(float(row.split(',')[3]))
            #precision_mc.append(float(row.split(',')[4]))
            #recall_mc.append(float(row.split(',')[5]))
            loss_mc.append(float(row.split(',')[3]))
            f1_mc.append(float(row.split(',')[4]))
            precision_mc.append(float(row.split(',')[5]))
            recall_mc.append(float(row.split(',')[6]))

    epoch_ml = []
    average_precision_ml = []
    loss_ml = []
    coverage_error_ml = []
    f1_ml = []
    precision_ml = []
    ranking_loss_ml = []
    recall_ml = []

    for row in multilabel_results:
        #print(row.split(','))
        epoch_ml.append(int(row.split(',')[0]))
        average_precision_ml.append(float(row.split(',')[1]))
        #coverage_error_ml.append(float(row.split(',')[2]))
        #f1_ml.append(float(row.split(',')[3]))
        #precision_ml.append(float(row.split(',')[4]))
        #ranking_loss_ml.append(float(row.split(',')[5]))
        #recall_ml.append(float(row.split(',')[6]))
        loss_ml.append(float(row.split(',')[2]))
        coverage_error_ml.append(float(row.split(',')[3]))
        f1_ml.append(float(row.split(',')[4]))
        precision_ml.append(float(row.split(',')[5]))
        ranking_loss_ml.append(float(row.split(',')[6]))
        recall_ml.append(float(row.split(',')[7]))
    
    return epoch_mc, accuracy_mc, balanced_accuracy_mc, loss_mc, f1_mc, precision_mc, recall_mc,\
    epoch_ml, average_precision_ml, loss_ml, coverage_error_ml, f1_ml, precision_ml, ranking_loss_ml, recall_ml
    
	
def plot_metrics(epoch_mc, accuracy_val_mc, accuracy_test_mc, epoch_ml, f1_val_ml, f1_test_ml, prec_val_ml, prec_test_ml, recall_val_ml, recall_test_ml, ontology_type):
    
    fig = plt.figure(figsize=(15,10))

    #fig.add_subplot(1,2,1)
    fig.add_subplot(2,2,1)
    plt.plot(epoch_mc,accuracy_val_mc, label='val')
    plt.plot(epoch_mc,accuracy_test_mc, label='test')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('MultiClass Accuracy')
    plt.grid(True)
    plt.legend()

    #fig.add_subplot(1,2,2)
    fig.add_subplot(2,2,2)
    plt.plot(epoch_ml,f1_val_ml, label='val')
    plt.plot(epoch_ml,f1_test_ml, label='test')
    plt.xlabel('epochs')
    plt.ylabel('f1-score')
    plt.title('MultiLabel F1-score')
    plt.grid(True)
    plt.legend()
    
    #fig.add_subplot(1,2,1)
    fig.add_subplot(2,2,3)
    plt.plot(epoch_mc,prec_val_ml, label='val')
    plt.plot(epoch_mc,prec_test_ml, label='test')
    plt.xlabel('epochs')
    plt.ylabel('precision')
    plt.title('MultiLabel Precision')
    plt.grid(True)
    plt.legend()

    #fig.add_subplot(1,2,2)
    fig.add_subplot(2,2,4)
    plt.plot(epoch_ml,recall_val_ml, label='val')
    plt.plot(epoch_ml,recall_test_ml, label='test')
    plt.xlabel('epochs')
    plt.ylabel('recall')
    plt.title('MultiLabel Recall')
    plt.legend()

    fig.suptitle(ontology_type, fontsize=20, y=1)
    fig.tight_layout()
    plt.grid(True)
	
	
	