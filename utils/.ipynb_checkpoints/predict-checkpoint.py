#import cntk
from keras.preprocessing import image
import keras
from keras.layers import Dense, Lambda, Activation, Input, Concatenate
from keras.models import load_model, Sequential
from keras import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras.applications.inception_resnet_v2 import InceptionResNetV2

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from training_wrapper.custom_layer import OntologyLayer

import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import confusion_matrix

#def predict_dishes(model, images_dir, prediction_file):    
#    
#    dish_lbls_pred = []
#    pred_directory = 'results/precision_dishes'
#    
#    if not os.path.exists(pred_directory):
#        os.makedirs(pred_directory)
#    
#   f = open(os.path.join(pred_directory,prediction_file),'w')

#   for root, subdirs, files in os.walk(images_dir):
#
#   if files:
#
#       for root_2, subdirs_2, files_2 in os.walk(root):
#       
#           for img_file in files_2:
#           
#                   img = image.load_img(os.path.join(root,img_file),target_size=(299,299,3))
#                   img = image.img_to_array(img)
#                   img = img/255
#                    img = np.expand_dims(img,axis=0)
                
#                    prediction = model.predict(img)
                
#                    SL_prediction = prediction[0][0]
                
#                    dish_lbls_pred.append(SL_prediction.argmax())
                    ##print(SL_prediction.argmax())
#                    f.write(str(SL_prediction.argmax())+'\n')

#    dish_lbls_pred = np.array(dish_lbls_pred)

def predict_dishes(model, images_dir, imgs_to_pred_file, pred_directory, prediction_file):    
    
    dish_lbls_pred = []
    images_to_pred = []
    
    if not os.path.exists(pred_directory):
        os.makedirs(pred_directory)
    
    pred_f = open(os.path.join(pred_directory,prediction_file),'w')
    
    with open(imgs_to_pred_file) as f:
        for line in f:
            images_to_pred.append(line.replace('\n',''))
                
    for img_file in images_to_pred: 
    
        #print(images_dir+img_file)
        img = image.load_img(images_dir+img_file,target_size=(299,299,3))
        img = image.img_to_array(img)
        img = img/255
        img = np.expand_dims(img,axis=0)
                
        prediction = model.predict(img)
                
        SL_prediction = prediction[0][0]
                
        dish_lbls_pred.append(SL_prediction.argmax())
        
        pred_f.write(str(SL_prediction.argmax())+'\n')

    dish_lbls_pred = np.array(dish_lbls_pred)
    
def predict_ingredients(model, images_dir, prob_confidence, imgs_to_pred_file, pred_directory, prediction_file, prediction_prob_file, out_of_confidence_predictions_file):    
    
    ingr_lbls_pred = []
    images_to_pred = []
    
    if not os.path.exists(pred_directory):
        os.makedirs(pred_directory)
    
    pred_f = open(os.path.join(pred_directory,prediction_file),'w')
    pred_prob_f = open(os.path.join(pred_directory,prediction_prob_file),'w')
    out_of_conf_f = open(os.path.join(pred_directory,out_of_confidence_predictions_file),'w')
    
    with open(imgs_to_pred_file) as f:
        for line in f:
            images_to_pred.append(line.replace('\n',''))
                
    for img_num, img_file in enumerate(images_to_pred): 
    
        img = image.load_img(images_dir+img_file,target_size=(299,299,3))
        img = image.img_to_array(img)
        img = img/255
        img = np.expand_dims(img,axis=0)
                
        prediction = model.predict(img)
                
        ML_prediction = prediction[1][0]
        
        #if the highest probability value is less than the confidence value, then take the thre most probable ingredients
        
        if ML_prediction.max() < prob_confidence:
            ingr_idx = ML_prediction.argsort()[::-1][:3]
            ingr_prob = ML_prediction[ingr_idx]
            out_of_conf_f.write(str(img_num)+'\n')
            
        else:
            ingr_idx = np.where(ML_prediction >= prob_confidence)[0]
            ingr_prob = ML_prediction[ingr_idx]
        
        ingr_idx = np.array(ingr_idx)
        ingr_prob = np.array(ingr_prob)
        
        
        for i,idx in enumerate(ingr_idx):
            if i == len(ingr_idx)-1:
                pred_f.write(str(idx)+'\n')
            else:
                pred_f.write(str(idx)+',')
        
        for i,prob in enumerate(ingr_prob):
            if i == len(ingr_prob)-1:
                pred_prob_f.write(str(prob)+'\n')
            else:
                pred_prob_f.write(str(prob)+',')
    
    
def built_model(n_dishes, n_ingredients, n_concepts, ontology_file, weights_file):
    
    pre_trained = InceptionResNetV2(InceptionResNetV2(weights='imagenet', input_shape=(299,299,3)))
    x = pre_trained.get_layer('avg_pool').output

    outputs = []

    out_SL_1 = Dense(n_dishes)(x)
    out_ML_1 = Dense(n_ingredients)(x)

    outputs.append(out_SL_1)
    outputs.append(out_ML_1)

    x = Concatenate()(outputs)
    x = OntologyLayer((None,n_concepts), ontology_file, False)(x)
    
    out_SL_2 = Lambda( lambda x: x[:, :n_dishes])(x)
    out_ML_2 = Lambda( lambda x: x[:, n_dishes:])(x)
    
    outputs_list = []
    
    out_SL_act = Activation('softmax')(out_SL_2)
    out_ML_act = Activation('sigmoid')(out_ML_2)
    
    outputs_list.append(out_SL_act)
    outputs_list.append(out_ML_act)
    
    model = Model(input=pre_trained.input,output=outputs_list)
    
    model.load_weights(weights_file)
    
    return model
	

images_dir = 'datasets/VireoFood172/images/'

def main(params):

    #print('Food Multitask with Dish Ingredient Probability Ontology')
    #model = built_model(params["n_dishes"], params["n_ingredients"], params["total_concepts"], params["ontology_list"][0], params["weights_list"][0])
    #prediction_file = 'pred_food_multitask_dish_ingr_ont.txt'
    #prediction_ingr_file = 'pred_food_multitask_dish_ingr_ont.txt'
    #prob_prediction_ingr_file = 'prob_pred_food_multitask_dish_ingr_ont.txt'
    #out_of_confidence_predictions_file = 'out_of_conf_pred_food_multitask_dish_ingr_ont.txt'

    #predict_dishes(model, params["images_dir"], params["imgs_to_pred_file"], params["pred_directory"], prediction_file)
    #predict_ingredients(model, params["images_dir"], params["prob_confidence"], params["imgs_to_pred_file"], params["pred_ingr_directory"], prediction_ingr_file,prob_prediction_ingr_file, out_of_confidence_predictions_file)
    
    #print('Food Multitask Baseline')
    #model = load_model('../models/food_multitask/epoch_9.h5')
    #prediction_file = 'pred_food_multitask.txt'
    #prediction_ingr_file = 'pred_food_multitask.txt'
    #prob_prediction_ingr_file = 'prob_pred_food_multitask.txt'
    #out_of_confidence_predictions_file = 'out_of_conf_pred_food_multitask.txt'

    #predict_dishes(model, params["images_dir"], params["imgs_to_pred_file"], params["pred_directory"], prediction_file)
    #predict_ingredients(model, params["images_dir"], params["prob_confidence"], params["imgs_to_pred_file"], params["pred_ingr_directory"], prediction_ingr_file,prob_prediction_ingr_file, out_of_confidence_predictions_file)
    
    #print('Food Multitask with Dish Ingredient Negative Probability Ontology')
    #model = built_model(params["n_dishes"], params["n_ingredients"], params["total_concepts"], params["ontology_list"][1], params["weights_list"][1])
    #prediction_file = 'pred_food_multitask_dish_ingr_ont_neg.txt'
    #prediction_ingr_file = 'pred_food_multitask_dish_ingr_ont_neg.txt'
    #prob_prediction_ingr_file = 'prob_pred_food_multitask_dish_ingr_ont_neg.txt'
    #out_of_confidence_predictions_file = 'out_of_conf_pred_food_multitask_dish_ingr_ont_neg.txt'

    #predict_dishes(model, params["images_dir"], params["imgs_to_pred_file"], params["pred_directory"], prediction_file)
    #predict_ingredients(model, params["images_dir"], params["prob_confidence"], params["imgs_to_pred_file"], params["pred_ingr_directory"], prediction_ingr_file,prob_prediction_ingr_file, out_of_confidence_predictions_file)

    print('Food Multitask with Dish Ingredient Ingredient Ingredient Negative Probability Ontology')
    model = built_model(params["n_dishes"], params["n_ingredients"], params["total_concepts"], params["ontology_list"][2], params["weights_list"][2])
    #prediction_file = 'pred_food_multitask_dish_ingr_ingr_ingr_ont_neg.txt'
    prediction_ingr_file = 'pred_food_multitask_dish_ingr_ingr_ingr_ont_neg.txt'
    prob_prediction_ingr_file = 'prob_pred_food_multitask_dish_ingr_ingr_ingr_ont_neg.txt'
    out_of_confidence_predictions_file = 'out_of_conf_pred_food_multitask_dish_ingr_ingr_ingr_ont_neg.txt'

    #predict_dishes(model, params["images_dir"], params["imgs_to_pred_file"], params["pred_directory"], prediction_file)
    predict_ingredients(model, params["images_dir"], params["prob_confidence"], params["imgs_to_pred_file"], params["pred_ingr_directory"], prediction_ingr_file,prob_prediction_ingr_file, out_of_confidence_predictions_file)

    print('Food Multitask with Ingredient Ingredient Negative Probability Ontology')
    model = built_model(params["n_dishes"], params["n_ingredients"], params["total_concepts"], params["ontology_list"][3], params["weights_list"][3])
    #prediction_file = 'pred_food_multitask_ingr_ingr_ont_neg.txt'
    prediction_ingr_file = 'pred_food_multitask_ingr_ingr_ont_neg.txt'
    prob_prediction_ingr_file = 'prob_pred_food_multitask_ingr_ingr_ont_neg.txt'
    out_of_confidence_predictions_file = 'out_of_conf_pred_food_multitask_ingr_ingr_ont_neg.txt'

    #predict_dishes(model, params["images_dir"], params["imgs_to_pred_file"], params["pred_directory"], prediction_file)
    predict_ingredients(model, params["images_dir"], params["prob_confidence"], params["imgs_to_pred_file"], params["pred_ingr_directory"], prediction_ingr_file,prob_prediction_ingr_file, out_of_confidence_predictions_file)


if __name__ == '__main__':
    params = {
        'images_dir' : '../datasets/VireoFood172/images/',
        'pred_directory' : '../results/precision_dishes',
        'pred_ingr_directory': '../results/precision_ingredients',
        'imgs_to_pred_file' : '../datasets/VireoFood172/TE.txt',
        'n_dishes' : 172,
        'n_ingredients' : 353,
        'total_concepts' : 525,
        'ontology_list' : [
                 '../datasets/VireoFood172/meta/ontology_files/Ontologies_prob/DI_ontology_matrix.npy',
                 '../datasets/VireoFood172/meta/ontology_files/Ontologies_prob_neg/DI_ontology_matrix.npy',
                 '../datasets/VireoFood172/meta/ontology_files/Ontologies_prob_neg/DI_II_ontology_matrix.npy',
                 '../datasets/VireoFood172/meta/ontology_files/Ontologies_prob_neg/II_ontology_matrix.npy'
                ],
        'weights_list' : [
                '../models/Ontology_probabilities_VIREO/Ontology_prob/food_multitask_dish_ingr_ont/epoch_9.h5',
                '../models/Ontology_probabilities_VIREO/Ontology_prob_neg/food_multitask_dish_ingr_ont/epoch_10.h5',
                '../models/Ontology_probabilities_VIREO/Ontology_prob_neg/food_multitask_dish_ingr_ingr_ingr_ont/epoch_10.h5',
                '../models/Ontology_probabilities_VIREO/Ontology_prob_neg/food_multitask_ingr_ingr_ont/epoch_10.h5'
               ],
        'prob_confidence' : 0.1
    }
	
main(params)





















