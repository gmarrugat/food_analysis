from keras_wrapper.dataset import Dataset, saveDataset, loadDataset

from collections import Counter
from operator import add

import nltk
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


def build_dataset(params, specific_params, empty_label=False):

    splits = ['train', 'val', 'test']
    #splits = ['train', 'test']
    if 'SORTED_OUTPUTS' in specific_params.keys():
        sorted_keys = specific_params['SORTED_OUTPUTS']
    else:
        sorted_keys = []
        for k in specific_params['OUTPUTS'].keys():
            if specific_params['OUTPUTS'][k]['type'] == 'sigma':
                sorted_keys.append(k)
            else:
                sorted_keys.insert(0, k)


    if specific_params['REBUILD_DATASET']: # We build a new dataset instance
        if(params['VERBOSE'] > 0):
            silence=False
            logging.info('Building ' + specific_params['DATASET_NAME'] + ' dataset')
        else:
            silence=True

        base_path = specific_params['DATA_ROOT_PATH']
        ds = Dataset(specific_params['DATASET_NAME'], base_path, silence=silence)

        ##### INPUT DATA
        for id_name, data in specific_params['INPUTS'].iteritems():
            if data['type'] == 'raw-image':
                for split in splits:
                    ds.setInput(base_path+'/'+data['path'] % split, split,
                               type=data['type'], id=id_name,
                               img_size=data['img_size'], img_size_crop=data['img_size_crop'])
            else:
                raise NotImplementedError('Input data type "'+data['type']+'" not implemented.')

        ##### OUTPUT DATA
        #for id_name, data in specific_params['OUTPUTS'].iteritems():
        print specific_params['OUTPUTS'].keys()
        for id_name in sorted_keys:
            data = specific_params['OUTPUTS'][id_name]
            if data['type'] == 'binary':
                [classes, word2idx, idx2word] = convertMultilabel2BinaryClasses(
                                                    base_path,
                                                    {split: data['path'] % split for split in splits},
                                                    data['classes'],
                                                    type_list_classes=data.get('labels_type_list', 'words'),
                                                    type_list=data.get('labels_type_list', 'identifiers'),
                                                    empty_label=empty_label,
                                                )
                for split in splits:
                    ds.setOutput(classes[split], split, type=data['type'],
                                 id=id_name, sparse=True)

                # Insert vocabularies
                ds.extra_variables['word2idx_'+id_name] = word2idx
                ds.extra_variables['idx2word_'+id_name] = idx2word

            elif data['type'] == 'categorical':
                for split in splits:
                    ds.setOutput(base_path+'/'+data['path'] % split, split,
                                 type=data['type'], id=id_name)

            elif data['type'] == 'sigma':
                pass # don't insert anything, just used for loss computation

            else:
                raise NotImplementedError('Output data type "'+data['type']+'" not implemented.')

        # Are we going to use class weights?
        class_weights = specific_params.get('CLASS_WEIGHTS', None)
        if class_weights is not None:

            # Read class weights files
            class_weights_dict = {}
            for o,f in class_weights.iteritems():
                class_weights_dict[o] = {}
                with open(base_path+'/'+f, 'r') as f:
                    for line in f:
                        line = line.rstrip('\n').split(',')
                        class_weights_dict[o][int(line[0])] = float(line[1])

            # Insert in dataset for later use
            ds.extra_variables['class_weights_'+specific_params['DATASET_NAME']] = class_weights_dict

        # We have finished loading the dataset, now we can store it for using it in the future
        saveDataset(ds, specific_params['DATA_ROOT_PATH'])


    else:
        # We can easily recover it with a single line
        ds = loadDataset(specific_params['DATA_ROOT_PATH']+'/Dataset_'+specific_params['DATASET_NAME']+'.pkl')


    ##### INPUTS and OUTPUTS mapping between Dataset and Model_Wrapper
    inputs_mapping = {}
    for i, id_name in enumerate(specific_params['INPUTS'].keys()):
        inputs_mapping[id_name] = i

    outputs_mapping = {}
    for i, id_name in enumerate(sorted_keys):
        # find matching output for sigma losses
        if specific_params['OUTPUTS'][id_name]['type'] == 'sigma':
            match_output = specific_params['OUTPUTS'][id_name]['output_id']
            i = sorted_keys.index(match_output)
        outputs_mapping[id_name] = i

    # Recover name of the class_weights variable if exists
    if 'class_weights_'+specific_params['DATASET_NAME'] in ds.extra_variables:
        class_weights = specific_params['DATASET_NAME']
    else:
        class_weights = None

    return ds, inputs_mapping, outputs_mapping, class_weights


def convertMultilabel2BinaryClasses(base_path, data, multilabels, type_list_classes='words', type_list='identifiers', empty_label=False):

    repeat_imgs = 1

    ing_list = []
    counter = Counter()
    with open(base_path+'/'+multilabels) as f:
        for pos_ing, line in enumerate(f):
            # read ingredients
            if type_list_classes == 'identifiers':
                ing = line.rstrip('\n').split(',')
                ing = map(lambda x: x.lower(), ing)
                ing_list.append(ing)
            elif type_list_classes == 'words':
                ing = line.rstrip('\n')
                ing = ing.lower()
                ing_list.append(ing)
                ing = [ing]
            counter.update(ing)

    # insert "empty" ingredient for samples without list of ingredients
    if empty_label:
        counter.update(['_empty_'])
        ing_list.append('_empty_')
        id_empty = len(ing_list)-1

    vocab_count = counter.most_common()

    vocabulary = {}
    list_words = []
    for i, (word, count) in enumerate(vocab_count):
        vocabulary[word] = i
        list_words.append(word)
    len_vocab = len(vocabulary)

    # Preprocess each data split
    classes = dict()
    for set_name, file in data.iteritems():
        classes[set_name] = []
        with open(base_path+'/'+file) as f:
            for idx_img, line in enumerate(f):
                #classes[set_name].append(np.zeros((len_vocab,)))
                if type_list=='identifiers':
                    #pos_ing = int(line.rstrip('\n'))
                    #ings = ing_list[pos_ing]
                    pos_ing = line.rstrip('\n').split(',')
                    if pos_ing[0] == '' and empty_label:
                        pos_ing = [str(id_empty)]
                    #ings = [ing_list[int(pos)] for pos in pos_ing]
                elif type_list=='words':
                    ings = line.rstrip('\n').split(',')

                classes[set_name].append(pos_ing)
                # insert all ingredients
                """
                for w in ings:
                    if w in vocabulary.keys():
                        classes[set_name][-1][vocabulary[w]] = 1
                """

    inv_vocabulary = {v: k for k, v in vocabulary.items()}
    return [classes, vocabulary, inv_vocabulary]
