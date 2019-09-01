import warnings
warnings.simplefilter("ignore",category=FutureWarning)

import logging
import numpy as np
from timeit import default_timer as timer
import copy
import sys

sys.path.insert(0,'/home/gerard/multimodal_keras_wrapper')
sys.path.insert(0,'/home/gerard/keras')
#sys.path.insert(0,'/media/HDD3TB/lib/LogMeal/multimodal_keras_wrapper')
#sys.path.insert(0,'/media/HDD3TB/lib/LogMeal/keras')

from keras_wrapper.cnn_model import saveModel, loadModel
from keras_wrapper.extra import evaluation
from keras_wrapper.extra.callbacks import EvalPerformance
from keras_wrapper.extra.read_write import *
from keras_wrapper.utils import decode_multilabel

from training_wrapper.general_params import load_parameters as load_general_parameters
from training_wrapper.prepare_data import build_dataset
from training_wrapper.model import Food_Model

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

from keras import backend as K


def train_model(params, specific_params):
    """
        Main function
    """

    ########### Load data
    if 'SPLITS' in specific_params.keys():
        dataset, inputs_mapping, outputs_mapping, class_weights = build_dataset(params, specific_params, empty_label=specific_params['EMPTY_LABEL'], splits=specific_params['SPLITS']) # dataset used for evaluation
    else:
        dataset, inputs_mapping, outputs_mapping, class_weights = build_dataset(params, specific_params, empty_label=specific_params['EMPTY_LABEL'])
    ###########

    ########### Build model
    params = mergeParams(params, specific_params)
    if params['RELOAD'] == 0:
        food_model = Food_Model(params,
                            type=specific_params['MODEL_TYPE'], verbose=params['VERBOSE'],
                            empty_label=params['EMPTY_LABEL'],
                            model_name=specific_params['MODEL_NAME'],
                            store_path=specific_params['STORE_PATH'])
    else:
        # Reload model previusly trained
        food_model = loadModel(params['STORE_PATH'], params['RELOAD'])

    # Define the inputs and outputs mapping from our Dataset instance to our model
    food_model.setInputsMapping(inputs_mapping)
    food_model.setOutputsMapping(outputs_mapping)

    # Update optimizer either if we are loading or building a model
    food_model.params = params
    food_model.setOptimizer()
    ###########


    ########### Callbacks
    callbacks = buildCallbacks(params, specific_params, food_model, dataset)
    ###########


    ########### Training
    total_start_time = timer()

    logger.debug('Starting training!')
    training_params = {'n_epochs': specific_params['MAX_EPOCH'], 'batch_size': specific_params['BATCH_SIZE'],
                       'lr_decay': specific_params['LR_DECAY'], 'lr_gamma': specific_params['LR_GAMMA'],
                       'epochs_for_save': params['EPOCHS_FOR_SAVE'], 'verbose': params['VERBOSE'],
                       'n_parallel_loaders': params['PARALLEL_LOADERS'],
                       'extra_callbacks': callbacks, 'reload_epoch': params['RELOAD'], 'epoch_offset': params['RELOAD'],
                       'data_augmentation': params['DATA_AUGMENTATION'],
                       'patience': specific_params['PATIENCE'], 'metric_check': specific_params['STOP_METRIC'],
                       'class_weights': class_weights,
                       }

    food_model.trainNet(dataset, training_params)

    total_end_time = timer()
    time_difference = total_end_time - total_start_time
    logging.info('Total time spent {0:.2f}s = {1:.2f}m'.format(time_difference, time_difference / 60.0))
    ###########




def buildCallbacks(params, specific_params, model, dataset):
    """
        Builds the selected set of callbacks run during the training of the model
    """

    callbacks = []

    if specific_params['EVALUATE']:

        if 'SORTED_OUTPUTS' in specific_params.keys():
            sorted_keys = specific_params['SORTED_OUTPUTS']
        else:
            sorted_keys = []
            for k in specific_params['OUTPUTS'].keys():
                if specific_params['OUTPUTS'][k]['type'] == 'sigma':
                    sorted_keys.append(k)
                else:
                    sorted_keys.insert(0, k)

        # Evaluate training
        extra_vars = dict()
        extra_vars['n_parallel_loaders'] = params['PARALLEL_LOADERS']

        #for i, (id_name, data) in enumerate(specific_params['OUTPUTS'].iteritems()):
        for i, id_name in enumerate(sorted_keys):
            data = specific_params['OUTPUTS'][id_name]
            if data['type'] != 'sigma':
                extra_vars[i] = dict()

        # Prepare GT for each output
        for s in params['EVAL_ON_SETS']:

            #for i, (id_name, data) in enumerate(specific_params['OUTPUTS'].iteritems()):
            for i, id_name in enumerate(sorted_keys):
                data = specific_params['OUTPUTS'][id_name]

                if data['type'] == 'binary':
                    extra_vars[i][s] = dict()
                    extra_vars[i][s]['word2idx'] = dataset.extra_variables['word2idx_'+id_name]
                elif data['type'] == 'categorical':
                    num_classes = 0
                    with open(specific_params['DATA_ROOT_PATH']+'/'+data['classes'], 'r') as f:
                        for line in f: num_classes += 1
                    extra_vars[i]['n_classes'] = num_classes
                    extra_vars[i][s] = dict()

        # Prepare other variables for performing evaluation
        gt_id_to_eval = []
        gt_pos_to_eval = []
        vocab = []
        vocab_x = []
        output_types = []
        metrics = []
        write_type = []
        min_pred_val = []
        #for i, (id_name, data) in enumerate(specific_params['OUTPUTS'].iteritems()):
        for i, id_name in enumerate(sorted_keys):
            data = specific_params['OUTPUTS'][id_name]
            if data['type'] == 'binary':
                vocab.append(dataset.extra_variables['idx2word_'+id_name])
            elif data['type'] == 'categorical':
                vocab.append('')
            elif data['type'] == 'sigma':
                break
            gt_id_to_eval.append(id_name)
            gt_pos_to_eval.append(i)
            vocab_x.append(None)
            output_types.append(data['type'])
            metrics.append(data['metrics'])
            write_type.append(data['write_type'])
            min_pred_val.append(data.get('min_pred_val', 0.5))

        callback_metric = EvalPerformance(model, dataset,
                                           model_name='model',
                                           inputs_mapping_eval=None,
                                           outputs_mapping_eval=None,
                                           gt_id=gt_id_to_eval,
                                           gt_pos=gt_pos_to_eval,
                                           metric_name=metrics,
                                           set_name=params['EVAL_ON_SETS'],
                                           batch_size=specific_params['BATCH_SIZE'],
                                           output_types=output_types,
                                           min_pred_multilabel=min_pred_val,
                                           index2word_y=vocab, # text info
                                           index2word_x=vocab_x,
                                           save_path=model.model_path,
                                           reload_epoch=params['RELOAD'],
                                           start_eval_on_epoch=params['START_EVAL_ON_EPOCH'],
                                           write_samples=params['WRITE_VALID_SAMPLES'],
                                           write_type=write_type,
                                           extra_vars=extra_vars,
                                           verbose=params['VERBOSE'],
                                           do_plot=False)
        callbacks.append(callback_metric)


    return callbacks

# Merges model-specific params to general params
def mergeParams(params, specific_params):
    for k,v in specific_params.iteritems():
        params[k] = v
    return params


if __name__ == "__main__":

    # Use 'config_file' command line parameter for changing the name of the config file used
    cf = 'train_config_v1'
    for arg in sys.argv[1:]:
        k, v = arg.split('=')
        if k == 'config_file':
            cf = v
    cf = __import__(cf)
    all_models_params = cf.load_parameters() # parameters for the current dataset version
    params = load_general_parameters() # general parameters


    # Configure GPU memory usage for TF
    if 'tensorflow' == K.backend():
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

    print "Training %d models" % len(all_models_params['ALL_MODELS'])

    # Train a model for each element in train_config_vX.py
    for specific_params in all_models_params['ALL_MODELS']:

        # Reset session (memory) before starting next network training
        if 'tensorflow' == K.backend():
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = params['GPU_MEMORY_FRACTION']
            config.gpu_options.visible_device_list = params['GPU_ID']
            config.gpu_options.allow_growth = True
            set_session(tf.Session(config=config))

        specific_params['STORE_PATH'] = specific_params['STORE_PATH'] % specific_params['MODELS_ROOT_PATH']
	print
        logging.info('======= Start training '+specific_params['MODEL_NAME']+' model =======')
        train_model(params, specific_params)
        logging.info('======= Finished training '+specific_params['MODEL_NAME']+' model =======')
	print


    logging.info('Done training all models!')
