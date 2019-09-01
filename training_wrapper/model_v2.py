from keras.engine import Input
from keras.layers import Average, Dropout, RepeatVector, Concatenate as Merge, Dense, Flatten, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.recurrent import LSTM
from keras.models import model_from_json, Sequential, Model
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

from keras.applications.resnet50 import ResNet50
#from keras.applications.resnext import ResNeXt50, ResNeXt101
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras_wrapper.cnn_model import Model_Wrapper
from keras_wrapper.cnn_model import loadModel
from custom_layer import ConcatenateOutputWithSigma, OntologyLayer
#from keras.layers import ConcatenateOutputWithSigma

from keras_models.keras_inceptionV4.inception_v4 import inception_v4

import numpy as np
import cPickle as pk
import os
import logging
import shutil
import time
import copy


class Food_Model(Model_Wrapper):

    def __init__(self, params, type='VGG16', verbose=1, structure_path=None, weights_path=None,
                 model_name=None, store_path=None, seq_to_functional=False,
                 empty_label=False):
        """
            Food_Model object constructor.

            :param params: all hyperparameters of the model.
            :param type: network name type (corresponds to any method defined in the section 'MODELS' of this class). Only valid if 'structure_path' == None.
            :param verbose: set to 0 if you don't want the model to output informative messages
            :param structure_path: path to a Keras' model json file. If we speficy this parameter then 'type' will be only an informative parameter.
            :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
            :param model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
            :param store_path: path to the folder where the temporal model packups will be stored
            :param seq_to_functional: defines if we are loading a set of weights from a Sequential model to a FunctionalAPI model (only applicable if weights_path is not None)
            :params empty_label: definies if we are introducing an "_empty_" label for the binary outputs
        """
        #super(self.__class__, self).__init__(type=type, model_name=model_name,
        #                                     silence=verbose == 0, models_path=store_path, inheritance=True)
        super(self.__class__, self).__init__(model_name=model_name,
                                             silence=verbose == 0, models_path=store_path, inheritance=True)
        self.__toprint = ['_model_type', 'name', 'model_path', 'verbose']

        self.verbose = verbose
        self._model_type = type
        self.params = params

        # Sets the model name and prepares the folders for storing the models
        self.setName(model_name, store_path)

        # Get sorted list of outputs
        if 'SORTED_OUTPUTS' in params.keys():
            sorted_keys = params['SORTED_OUTPUTS']
        else:
            sorted_keys = []
            for k in params['OUTPUTS'].keys():
                if params['OUTPUTS'][k]['type'] == 'sigma':
                    sorted_keys.append(k)
                else:
                    sorted_keys.insert(0, k)

        # Prepare lists of labels and store them for decoding predictions
        self.labels_list = {}
        for id_out in sorted_keys:
            data = self.params['OUTPUTS'][id_out]
            self.labels_list[id_out] = {'labels': []}
            self.labels_list[id_out]['type'] = data['type']
            self.labels_list[id_out]['min_pred_val'] = data.get('min_pred_val', 0.5)
            if data['type'] != 'sigma': # only ready GT labels if output is not of sigma type
                with open(self.params['DATA_ROOT_PATH']+'/'+data['classes'], 'r') as fc:
                    for line in fc:
                        self.labels_list[id_out]['labels'].append(line.rstrip('\n'))
                    if data['type'] == 'binary' and empty_label:
                        self.labels_list[id_out]['labels'].append('_empty_')
            logging.info(self.labels_list[id_out]['labels']);

        # Prepare model
        if structure_path:
            # Load a .json model
            if self.verbose > 0:
                logging.info("<<< Loading model structure from file "+ structure_path +" >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, type):
                if self.verbose > 0:
                    logging.info("<<< Building "+ type +" Food_Model >>>")
                eval('self.'+type+'(params)')
            else:
                raise Exception('Food_Model type "'+ type +'" is not implemented.')

        # Load weights from file
        if weights_path:
            if self.verbose > 0:
                logging.info("<<< Loading weights from file "+ weights_path +" >>>")
            self.model.load_weights(weights_path, seq_to_functional=seq_to_functional)

        # Print information of self
        if verbose > 0:
            print str(self)
            self.model.summary()

        self.setOptimizer()


    def setOptimizer(self, metrics=['acc']):

        """
            Sets a new optimizer for the model.
        """
        uncertainty_loss = self.params.get('USE_UNCERTAINTY_LAYER', False)
        if 'SORTED_OUTPUTS' in self.params.keys():
            sorted_keys = self.params['SORTED_OUTPUTS']
        else:
            sorted_keys = []
            for k in self.params['OUTPUTS'].keys():
                if self.params['OUTPUTS'][k]['type'] == 'sigma':
                    sorted_keys.append(k)
                else:
                    sorted_keys.insert(0, k)
	loss = [self.params['OUTPUTS'][k]['loss'] for k in sorted_keys]

        super(self.__class__, self).setOptimizer(lr=self.params['LR'],
                                                 loss=loss if not uncertainty_loss else None,
                                                 optimizer=self.params['OPTIMIZER'],
                                                 loss_weights=self.params.get('LOSS_WIGHTS', None),
                                                 sample_weight_mode='temporal' if self.params.get('SAMPLE_WEIGHTS', False) else None)


    def setName(self, model_name, store_path=None, clear_dirs=True):
        """
            Changes the name (identifier) of the model instance.
        """
        if model_name is None:
            self.name = time.strftime("%Y-%m-%d") + '_' + time.strftime("%X")
            create_dirs = False
        else:
            self.name = model_name
            create_dirs = True

        if store_path is None:
            self.model_path = 'Models/' + self.name
        else:
            self.model_path = store_path


        # Remove directories if existed
        if clear_dirs:
            if os.path.isdir(self.model_path):
                shutil.rmtree(self.model_path)

        # Create new ones
        if create_dirs:
            if not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)


    def predict_and_decode(self, X):
        '''
            Applies a forward pass on the samples loaded in 'X' and applies
            the decoding of all the outputs provided by the network.

            Returns list with an element per image, where each element is a
            dictionary containing the following information:
                    {
		        'label1': {
                                      'labels': ['string1', 'string2', ..., 'stringN'],
                                      'probs': [prob1, prob2, ..., probN],
                                  },
                        'label2': {
                                      'labels': ['string1', 'string2', ..., 'stringM'],
                                      'probs': [prob1, prob2, ..., probM],
                                  },
                    }
        '''

        # Predict output
        preds = self.predictOnBatch(X)

        if 'SORTED_OUTPUTS' in self.params.keys():
            sorted_keys = self.params['SORTED_OUTPUTS']
        else:
            sorted_keys = []
            for k in self.params['OUTPUTS'].keys():
                if self.params['OUTPUTS'][k]['type'] == 'sigma':
                    sorted_keys.append(k)
                else:
                    sorted_keys.insert(0, k)

        n_outputs = len(sorted_keys)
        if n_outputs == 1:
            n_samples = len(preds)
        else:
            n_samples = preds[0].shape[0]

        # Decode
        decodification = [{} for im in range(n_samples)]
        for k,id in enumerate(sorted_keys):
            #for k,(id,pos) in enumerate(self.outputsMapping.iteritems()):
            pos = self.outputsMapping[id]
            type = self.labels_list[id]['type']
            if type != 'sigma':
                labels = self.labels_list[id]['labels']
                min_pred_val = self.labels_list[id]['min_pred_val']

                for im in range(n_samples): # decodify each image
                    if n_outputs == 1:
                        these_preds = preds[im]
                    else:
                        these_preds = preds[pos][im]

                    if k == 0:
                        decodification[im] = {}
                    decodification[im][id] = {'labels': [], 'probs': []}

                    if type == 'categorical':
                        for i,prob in enumerate(these_preds):
                            decodification[im][id]['labels'].append(labels[i])
                            decodification[im][id]['probs'].append(float(prob))
                    elif type == 'binary':
                        for i,prob in enumerate(these_preds):
                            if prob >= min_pred_val:
                                decodification[im][id]['labels'].append(labels[index2word[i]])
                                decodification[im][id]['probs'].append(float(prob))
                    else:
                        raise NotImplementedError("Decodification for type '"+type+"' is not implemented.")

        return decodification

    # ------------------------------------------------------- #
    #       VISUALIZATION
    #           Methods for visualization
    # ------------------------------------------------------- #

    def __str__(self):
        """
            Plot basic model information.
        """
        obj_str = '-----------------------------------------------------------------------------------\n'
        class_name = self.__class__.__name__
        obj_str += '\t\t'+class_name +' instance\n'
        obj_str += '-----------------------------------------------------------------------------------\n'

        # Print pickled attributes
        for att in self.__toprint:
            obj_str += att + ': ' + str(self.__dict__[att])
            obj_str += '\n'

        obj_str += '\n'
        obj_str += 'MODEL PARAMETERS:\n'
        obj_str += str(self.params)
        obj_str += '\n'

        obj_str += '-----------------------------------------------------------------------------------'

        return obj_str


    # ------------------------------------------------------- #
    #       PREDEFINED MODELS
    # ------------------------------------------------------- #

    def ResNet50_PlusFC(self, params):

        assert len(params['INPUTS'].keys()) == 1, 'Number of inputs must be one.'
        assert params['INPUTS'][params['INPUTS'].keys()[0]]['type'] == 'raw-image', 'Input must be of type "raw-image".'

        self.ids_inputs = params['INPUTS'].keys()
        self.ids_outputs = params['OUTPUTS'].keys()

        input_shape = params['INPUTS'][params['INPUTS'].keys()[0]]['img_size_crop']
        image = Input(name=self.ids_inputs[0], shape=input_shape)

        ##################################################
        # Load Inception model pre-trained on ImageNet
        self.model = ResNet50(weights='imagenet', input_tensor=image)

        # Recover input layer
        #image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model: 'fc2'
        x = self.model.get_layer('avg_pool').output
        ##################################################
        #x = Flatten()(x)
        # Define outputs
        outputs_list = []
        for id_name, data in params['OUTPUTS'].iteritems():

            # Count the number of output classes
            num_classes = 0
            with open(params['DATA_ROOT_PATH']+'/'+data['classes'], 'r') as f:
                for line in f: num_classes += 1

            # Define only a FC output layer (+ activation) per output
            out = Dense(num_classes)(x)
            out_act = Activation(data['activation'], name=id_name)(out)
            outputs_list.append(out_act)


        self.model = Model(input=image, output=outputs_list)


    def InceptionV3_PlusFC(self, params):

        assert len(params['INPUTS'].keys()) == 1, 'Number of inputs must be one.'
        assert params['INPUTS'][params['INPUTS'].keys()[0]]['type'] == 'raw-image', 'Input must be of type "raw-image".'

        self.ids_inputs = params['INPUTS'].keys()
        self.ids_outputs = params['OUTPUTS'].keys()

        input_shape = params['INPUTS'][params['INPUTS'].keys()[0]]['img_size_crop']
        image = Input(name=self.ids_inputs[0], shape=input_shape)

        ##################################################
        # Load Inception model pre-trained on ImageNet
        self.model = InceptionV3(weights='imagenet', input_tensor=image)

        # Recover input layer
        #image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model: 'fc2'
        x = self.model.get_layer('avg_pool').output
        ##################################################
        #x = Flatten()(x)
        # Define outputs
        outputs_list = []
        for id_name, data in params['OUTPUTS'].iteritems():

            # Count the number of output classes
            num_classes = 0
            with open(params['DATA_ROOT_PATH']+'/'+data['classes'], 'r') as f:
                for line in f: num_classes += 1

            # Define only a FC output layer (+ activation) per output
            out = Dense(num_classes)(x)
            out_act = Activation(data['activation'], name=id_name)(out)
            outputs_list.append(out_act)


        self.model = Model(input=image, output=outputs_list)


    def InceptionResNetV2_PlusFC(self, params):

        assert len(params['INPUTS'].keys()) == 1, 'Number of inputs must be one.'
        assert params['INPUTS'][params['INPUTS'].keys()[0]]['type'] == 'raw-image', 'Input must be of type "raw-image".'

        self.ids_inputs = params['INPUTS'].keys()
        self.ids_outputs = params['OUTPUTS'].keys()

        input_shape = params['INPUTS'][params['INPUTS'].keys()[0]]['img_size_crop']
        image = Input(name=self.ids_inputs[0], shape=input_shape)

        ##################################################
        # Load Inception model pre-trained on ImageNet
        self.model = InceptionResNetV2(weights='imagenet', input_tensor=image)

        # Recover input layer
        #image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model: 'fc2'
        x = self.model.get_layer('avg_pool').output
        ##################################################
        #x = Flatten()(x)
        # Define outputs
        outputs_list = []
        for id_name, data in params['OUTPUTS'].iteritems():

            # Count the number of output classes
            num_classes = 0
            with open(params['DATA_ROOT_PATH']+'/'+data['classes'], 'r') as f:
                for line in f: num_classes += 1
            if params['EMPTY_LABEL']:
                num_classes += 1
            # Define only a FC output layer (+ activation) per output
            out = Dense(num_classes)(x)
            out_act = Activation(data['activation'], name=id_name)(out)
            outputs_list.append(out_act)


        self.model = Model(input=image, output=outputs_list)

   
    def InceptionResNetV2_PlusFC_LMW(self, params):

        assert len(params['INPUTS'].keys()) == 1, 'Number of inputs must be one.'
        assert params['INPUTS'][params['INPUTS'].keys()[0]]['type'] == 'raw-image', 'Input must be of type "raw-image".'

        self.ids_inputs = params['INPUTS'].keys()
        self.ids_outputs = params['OUTPUTS'].keys()

        model_folder = '/home/eduardo/Documents/workspaces/ifood2019_recognition/models/inception_resnet_v2_logmealv3/models'
        model_reload_epoch = 9
        # Load model
        base_model = loadModel(model_folder, model_reload_epoch).model
        ##################################################
        x = base_model.get_layer('avg_pool').output
        ##################################################
        # Define outputs
        outputs_list = []
        for id_name, data in params['OUTPUTS'].iteritems():
            # Count the number of output classes
            num_classes = 0
            with open(params['DATA_ROOT_PATH']+'/'+data['classes'], 'r') as f:
                for line in f: num_classes += 1
            if params['EMPTY_LABEL']:
                num_classes += 1
            # Define only a FC output layer (+ activation) per output
            out = Dense(num_classes)(x)
            out_act = Activation(data['activation'], name=id_name)(out)
            outputs_list.append(out_act)

        self.model = Model(input=base_model.input, output=outputs_list)


    def InceptionResNetV2_Multitask(self, params):

        assert len(params['INPUTS'].keys()) == 1, 'Number of inputs must be one.'
        assert params['INPUTS'][params['INPUTS'].keys()[0]]['type'] == 'raw-image', 'Input must be of type "raw-image".'

        self.ids_inputs = params['INPUTS'].keys()
        self.ids_outputs = params['OUTPUTS'].keys()

        input_shape = params['INPUTS'][params['INPUTS'].keys()[0]]['img_size_crop']
        image = Input(name=self.ids_inputs[0], shape=input_shape)

        ##################################################
        # Load Inception model pre-trained on ImageNet
        self.model = InceptionResNetV2(weights='imagenet', input_tensor=image)
		
        for layer in self.model.layers:
	        layer.trainable=False
			

        # Recover input layer
        #image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model: 'fc2'
        x = self.model.get_layer('avg_pool').output
        #x = self.model.output
		##################################################
        #x = Flatten()(x)
        # Define outputs
        outputs_list = []
        outputs_matching = {}
        num_classes_matching = {}

        if 'SORTED_OUTPUTS' in params.keys():
            sorted_keys = params['SORTED_OUTPUTS']
        else:
            sorted_keys = []
            for k in params['OUTPUTS'].keys():
                if params['OUTPUTS'][k]['type'] == 'sigma':
                    sorted_keys.append(k)
                else:
                    sorted_keys.insert(0, k)

        #for id_name, data in params['OUTPUTS'].iteritems():
        for id_name in sorted_keys:
            data = params['OUTPUTS'][id_name]

            # Special output that calculates sigmas for uncertainty loss
            if data['type'] == 'sigma':
                match_output = params['OUTPUTS'][id_name]['output_id']
                match_act = outputs_matching[match_output]

                out_sigma = ConcatenateOutputWithSigma((None, num_classes_matching[match_output]+1), name_suffix=id_name, name=id_name)(match_act)
                outputs_list.append(out_sigma)

            else:
                # Count the number of output classes
                num_classes = 0
                with open(params['DATA_ROOT_PATH']+'/'+data['classes'], 'r') as f:
                    for line in f: num_classes += 1
                if data['type'] == 'binary' and params['EMPTY_LABEL']==True:
                    num_classes += 1 # empty label

                # Define only a FC output layer (+ activation) per output
                out = Dense(num_classes)(x)
                out_act = Activation(data['activation'], name=id_name)(out)
                outputs_list.append(out_act)

                outputs_matching[id_name] = out_act
                num_classes_matching[id_name] = num_classes


        self.model = Model(input=image, output=outputs_list)


    def InceptionResNetV2_Multitask_Ont(self, params):

        assert len(params['INPUTS'].keys()) == 1, 'Number of inputs must be one.'
        assert params['INPUTS'][params['INPUTS'].keys()[0]]['type'] == 'raw-image', 'Input must be of type "raw-image".'

        self.ids_inputs = params['INPUTS'].keys()
        self.ids_outputs = params['OUTPUTS'].keys()

        input_shape = params['INPUTS'][params['INPUTS'].keys()[0]]['img_size_crop']
        image = Input(name=self.ids_inputs[0], shape=input_shape)

        ##################################################
        # Load Inception model pre-trained on ImageNet
        self.model = InceptionResNetV2(weights='imagenet', input_tensor=image)

        # Recover input layer
        #image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model: 'fc2'
        x = self.model.get_layer('avg_pool').output
        ##################################################
        #x = Flatten()(x)
        # Define outputs
        outputs = []
        outputs_list = []
        outputs_matching = {}
        num_classes_matching = {}

        if 'SORTED_OUTPUTS' in params.keys():
            sorted_keys = params['SORTED_OUTPUTS']
        else:
            sorted_keys = []
            for k in params['OUTPUTS'].keys():
                if params['OUTPUTS'][k]['type'] == 'sigma':
                    sorted_keys.append(k)
                else:
                    sorted_keys.insert(0, k)
        num_classes_list = []
        for id_name in sorted_keys:
            data = params['OUTPUTS'][id_name]
            if data['type'] == 'sigma':
                continue
            else:
                # Count the number of output classes
                num_classes = 0
                with open(params['DATA_ROOT_PATH']+'/'+data['classes'], 'r') as f:
                    for line in f: num_classes += 1
                if data['type'] == 'binary' and params['EMPTY_LABEL']==True:
                    num_classes += 1 # empty label

                # Define only a FC output layer per output
                out = Dense(num_classes)(x)
                outputs.append(out)
                num_classes_list.append(num_classes)

        n_multiclass = num_classes_list[0]
        n_multilabel = num_classes_list[1]
        total_concepts = np.sum(num_classes_list)
        x = Merge()(outputs)

        #Ont_Layer = OntologyLayer((None,total_concepts),"/media/HDD3TB/datasets/Recipes5k/mtannotations/Ontology_matrix.npy")
        #Ont_Layer.build((None,n_multiclass))
        #Ont_Layer.trainable = False

        x = OntologyLayer((None,total_concepts),"/media/HDD3TB/datasets/Recipes5k/mtannotations/Ontology_matrix.npy")(x)
        
        outputs = []

        for idx, num_classes  in enumerate(num_classes_list):
            if idx == 0:
                init_idx = 0
                end_idx = num_classes
            else:
                init_idx = end_idx
                end_idx = end_idx + num_classes
            out = Lambda( lambda x: x[:, init_idx:end_idx])(x)
            outputs.append(out)

        curr_output = 0
        for id_name in sorted_keys:
            data = params['OUTPUTS'][id_name]
            # Special output that calculates sigmas for uncertainty loss
            if data['type'] == 'sigma':
                match_output = params['OUTPUTS'][id_name]['output_id']
                match_act = outputs_matching[match_output]
                out_sigma = ConcatenateOutputWithSigma((None, num_classes_matching[match_output]+1), name_suffix=id_name, name=id_name)(match_act)
                outputs_list.append(out_sigma)
            else:
                out = outputs[curr_output]
                curr_output = curr_output + 1
                out_act = Activation(data['activation'], name=id_name)(out)
                outputs_list.append(out_act)
                outputs_matching[id_name] = out_act
                num_classes_matching[id_name] = num_classes


        self.model = Model(input=image, output=outputs_list)
	def InceptionResNetV2_Multitask_TopDown_Ont(self, params):

        assert len(params['INPUTS'].keys()) == 1, 'Number of inputs must be one.'
        assert params['INPUTS'][params['INPUTS'].keys()[0]]['type'] == 'raw-image', 'Input must be of type "raw-image".'

        self.ids_inputs = params['INPUTS'].keys()
        self.ids_outputs = params['OUTPUTS'].keys()

        input_shape = params['INPUTS'][params['INPUTS'].keys()[0]]['img_size_crop']
        image = Input(name=self.ids_inputs[0], shape=input_shape)

        ##################################################
        # Load Inception model pre-trained on ImageNet
        self.model = InceptionResNetV2(weights='imagenet', input_tensor=image)

        # Recover input layer
        #image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model: 'fc2'
        x = self.model.get_layer('avg_pool').output
        ##################################################
        #x = Flatten()(x)
        # Define outputs
        outputs = []
        outputs_list = []
        outputs_matching = {}
        num_classes_matching = {}

        if 'SORTED_OUTPUTS' in params.keys():
            sorted_keys = params['SORTED_OUTPUTS']
        else:
            sorted_keys = []
            for k in params['OUTPUTS'].keys():
                if params['OUTPUTS'][k]['type'] == 'sigma':
                    sorted_keys.append(k)
                else:
                    sorted_keys.insert(0, k)
        num_classes_list = []
        for id_name in sorted_keys:
            data = params['OUTPUTS'][id_name]
            if data['type'] == 'sigma':
                continue
            else:
                # Count the number of output classes
                num_classes = 0
                with open(params['DATA_ROOT_PATH']+'/'+data['classes'], 'r') as f:
                    for line in f: num_classes += 1
                if data['type'] == 'binary' and params['EMPTY_LABEL']==True:
                    num_classes += 1 # empty label

                # Define only a FC output layer per output
                out = Dense(num_classes)(x)
                outputs.append(out)
                num_classes_list.append(num_classes)

		n_multiclass = num_classes_list[0]
		n_multilabel = num_classes_list[1]
        total_concepts = np.sum(num_classes_list)
        x = Merge()(outputs)
        x = OntologyLayer((None,total_concepts), "/media/HDD3TB/datasets/Recipes5k/mtannotations/Ontology_matrix.npy")(x)
        
        outputs = []
			
	for idx, num_classes  in enumerate(num_classes_list):
            if idx == 0:
                init_idx = 0
                end_idx = num_classes
            else:
                init_idx = end_idx
                end_idx = end_idx + num_classes
            out = Lambda( lambda x: x[:, init_idx:end_idx])(x)
            outputs.append(out)

	curr_output = 0
        for id_name in sorted_keys:
            data = params['OUTPUTS'][id_name]
            # Special output that calculates sigmas for uncertainty loss
            if data['type'] == 'sigma':
                match_output = params['OUTPUTS'][id_name]['output_id']
                match_act = outputs_matching[match_output]
                out_sigma = ConcatenateOutputWithSigma((None, num_classes_matching[match_output]+1), name_suffix=id_name, name=id_name)(match_act)
                outputs_list.append(out_sigma)
            else:
                out = outputs[curr_output]
                curr_output = curr_output + 1
                out_act = Activation(data['activation'], name=id_name)(out)
                outputs_list.append(out_act)
                outputs_matching[id_name] = out_act
                num_classes_matching[id_name] = num_classes


        self.model = Model(input=image, output=outputs_list)

    def InceptionV4_PlusFC(self, params):

        assert len(params['INPUTS'].keys()) == 1, 'Number of inputs must be one.'
        assert params['INPUTS'][params['INPUTS'].keys()[0]]['type'] == 'raw-image', 'Input must be of type "raw-image".'

        self.ids_inputs = params['INPUTS'].keys()
        self.ids_outputs = params['OUTPUTS'].keys()

        input_shape = params['INPUTS'][params['INPUTS'].keys()[0]]['img_size_crop']
        image = Input(name=self.ids_inputs[0], shape=input_shape)

        ##################################################
        # Load Inception model pre-trained on ImageNet
        #self.model = InceptionResNetV2(weights='imagenet', input_tensor=image)
        self.model = inception_v4(251, 0.5, "imagenet", False)
        # Recover input layer
        #image = self.model.get_layer(self.ids_inputs[0]).output
        # Recover last layer kept from original model: 'fc2'
        x = self.model.get_layer('concatenate_25').output
        ##################################################
        # 1 x 1 x 1536
        x = AveragePooling2D((8,8), padding='valid')(x)
        #x = Dropout(0.5)(x)
        x = Flatten()(x)
        # 1536

        # Define outputs
        outputs_list = []
        for id_name, data in params['OUTPUTS'].iteritems():
            # Count the number of output classes
            num_classes = 0
            with open(params['DATA_ROOT_PATH']+'/'+data['classes'], 'r') as f:
                for line in f: num_classes += 1
            if params['EMPTY_LABEL']:
                num_classes += 1
            # Define only a FC output layer (+ activation) per output
            out = Dense(num_classes)(x)
            out_act = Activation(data['activation'], name=id_name)(out)
            outputs_list.append(out_act)


        self.model = Model(input=self.model.input, output=outputs_list)

    '''
    def ResNext50_PlusFC(self, params):

        assert len(params['INPUTS'].keys()) == 1, 'Number of inputs must be one.'
        assert params['INPUTS'][params['INPUTS'].keys()[0]]['type'] == 'raw-image', 'Input must be of type "raw-image".'

        self.ids_inputs = params['INPUTS'].keys()
        self.ids_outputs = params['OUTPUTS'].keys()

        input_shape = params['INPUTS'][params['INPUTS'].keys()[0]]['img_size_crop']
        image = Input(name=self.ids_inputs[0], shape=input_shape)

        ##################################################
        # Load Inception model pre-trained on ImageNet
        self.model = ResNeXt50(weights='imagenet', input_tensor=image)

        # Recover input layer
        #image = self.model.get_layer(self.ids_inputs[0]).output

        # Recover last layer kept from original model: 'fc2'
        x = self.model.get_layer('avg_pool').output
        ##################################################
        #x = Flatten()(x)
        # Define outputs
        outputs_list = []
        for id_name, data in params['OUTPUTS'].iteritems():
            # Count the number of output classes
            num_classes = 0
            with open(params['DATA_ROOT_PATH']+'/'+data['classes'], 'r') as f:
                for line in f: num_classes += 1
            # Define only a FC output layer (+ activation) per output
            out = Dense(num_classes)(x)
            out_act = Activation(data['activation'], name=id_name)(out)
            outputs_list.append(out_act)


        self.model = Model(input=image, output=outputs_list)

    '''

    def InceptionResNetV2_Ensemble(self, params):

        #assert len(params['INPUTS'].keys()) == 1, 'Number of inputs must be one.'
        #assert params['INPUTS'][params['INPUTS'].keys()[0]]['type'] == 'raw-image', 'Input must be of type "raw-image".'

        self.ids_inputs = params['INPUTS'].keys()
        self.ids_outputs = params['OUTPUTS'].keys()
        input_shape = params['INPUTS'][params['INPUTS'].keys()[0]]['img_size_crop']
        image = Input(name=self.ids_inputs[0], shape=input_shape)


        models_folder = ['inceptionresnetv2_LMW_adam_1', 
                     'inceptionresnetv2_LMW_adam_2', 
                     'inceptionresnetv2_LMW_adam_3',
                     'inceptionresnetv2_LMW_adam_4',
                     'inceptionresnetv2_LMW_adam_5',
                     'inceptionresnetv2_LMW_adam_6',
                     'inceptionresnetv2_LMW_adam_7']
        models_reload_epoch = [9, 
                           13, 
                           14, 
                           11, 
                           17,
                           10,
                           12]

        models = []

        for idx, bmodel_folder in enumerate(models_folder):
            model_folder = '/home/eduardo/Documents/workspaces/ifood2019_recognition/models/'+bmodel_folder
            model_reload_epoch = models_reload_epoch[idx]
            # Load model
            base_model = loadModel(model_folder, model_reload_epoch).model
            base_model.name = "emodel_"+str(idx)
            models.append(base_model)

        #models = [base_model_1, base_model_2]
        merged_models = []
        for j in range(len(models)):
            base_model = models[j]
            for i, layer in enumerate(base_model.layers[1:]):
                layer.trainable = False
                print layer.name
            merged_models.append(base_model(image))

        x = Merge()(merged_models)
        ##################################################
        # Define outputs
        outputs_list = []
        for id_name, data in params['OUTPUTS'].iteritems():
            # Count the number of output classes
            num_classes = 0
            with open(params['DATA_ROOT_PATH']+'/'+data['classes'], 'r') as f:
                for line in f: num_classes += 1
            if data['type'] == 'binary':
                num_classes += 1 # empty label

            # Define only a FC output layer (+ activation) per output
            x = Dense(num_classes*len(models), activation="relu")(x)
            out = Dense(num_classes, kernel_initializer="ones")(x)
            out_act = Activation(data['activation'], name=id_name)(out)
            outputs_list.append(out_act)
        
        self.model = Model(input=image, output=outputs_list)


