#from training_wrapper.custom_losses import uncertainty_categorical_crossentropy, uncertainty_binary_crossentropy, regularizer_mtl_loss,deactivate_loss
from keras.layers import uncertainty_categorical_crossentropy, uncertainty_binary_crossentropy, deactivate_loss

def load_parameters():
    """
        Loads the defined parameters
    """

    # Train non_food vs food vs drinks vs ingredients model
    ALL_MODELS = [
                        ###################
                        # Food only
                        ###################
                        {
                        # Model params
                        'MODEL_TYPE': 'InceptionResNetV2_Multitask_Ont', # see predefined models in models.py
                        'MODEL_NAME': 'Food_multitask',
                        'STORE_PATH': '%s/food_multitask_ingr_dish_ont', # insert MODELS_ROOT_PATH in %s
                        'MODELS_ROOT_PATH': 'models/Ontology_1_0/',
                        'EVALUATE': True,
                        # Dataset params
                        'DATA_ROOT_PATH': '/media/HDD3TB/gerard/food_analysis-dag_labels/datasets/Recipes5k',#'/home/eduardo/Documents/datasets/Recipes5k/',
                        'DATASET_NAME': 'Food_Multitask',
                        'REBUILD_DATASET': True,
                        'ONTOLOGY': '/media/HDD3TB/gerard/food_analysis-dag_labels/datasets/Recipes5k/mtannotations/Ontologies_1_0/ID_ontology_matrix.npy',
                        'INPUTS': {'images': {
                                            'type': 'raw-image',
                                            'path': 'mtannotations/list_%s_img.txt',
                                            'img_size': [342, 342, 3],
                                            'img_size_crop': [299, 299, 3],
                                            },
                        },
                        'OUTPUTS': {'dish': {
                                            'type': 'categorical',
                                            'path': 'mtannotations/list_dish_%s_img.txt',
                                            'classes': 'mtannotations/dish_list.txt',
                                            'metrics': ['multiclass_metrics'], # multilabel_metrics or multiclass_metrics
                                            'write_type': 'list', # listoflists or list
                                            'activation': 'softmax', # sigmoid or softmax
                                            'loss': 'categorical_crossentropy', # binary_crossentropy or categorical_crossentropy
                                            #'loss': deactivate_loss(), # binary_crossentropy or categorical_crossentropy
                                            },
                                    #'dish_sigma': {
                                    #        'type': 'sigma',
                                    #        'output_id': 'dish', # name of the output linked to this sigma
                                    #        'loss': uncertainty_categorical_crossentropy(True, True, [418]), # binary_crossentropy or categorical_crossentropy
                                    #        },
                                    'categories': {
                                            'type': 'binary',
                                            'path': 'mtannotations/list_idx_ingredients_%s_img_f.txt',
                                            'classes': 'mtannotations/ingredients_list.txt',
                                            'metrics': ['multilabel_metrics'], # multilabel_metrics or multiclass_metrics
                                            'write_type': 'list', # listoflists or list
                                            'activation': 'sigmoid', # sigmoid or softmax
                                            'loss': 'binary_crossentropy', # binary_crossentropy or categorical_crossentropy
                                            #'loss': deactivate_loss(), # binary_crossentropy or categorical_crossentropy
                                            'min_pred_val': 0.5, # only used for outputs of type 'binary'
                                            },
                                    #'categories_sigma': {
                                    #        'type': 'sigma',
                                    #        'output_id': 'categories', # name of the output linked to this sigma
                                    #        'loss': uncertainty_binary_crossentropy(True, True, [15]), # binary_crossentropy or categorical_crossentropy
                                    #        },
                        },
                        #'SORTED_OUTPUTS': ['dish', 'categories', 'dish_sigma', 'categories_sigma'],
                        'SORTED_OUTPUTS': ['dish', 'categories'],
                        # Training params
                        'EMPTY_LABEL': False,
                        'RELOAD': 0,
                        'PATIENCE': 50,
                        'STOP_METRIC': 'f1_output_0',
                        'MAX_EPOCH': 50,
                        'BATCH_SIZE': 8,
                        'LR_DECAY': None, # number of minimum number of epochs before the next LR decay (set to None to disable)
                        'LR_GAMMA': 0.5, # multiplier used for decreasing the LR
                        },

    ]

    #ALL_MODELS = ALL_MODELS[0]

    # ============================================
    parameters = locals().copy()
    return parameters
