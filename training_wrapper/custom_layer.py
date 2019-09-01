from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class ConcatenateOutputWithSigma(Layer):

    def __init__(self, output_dim, name_suffix, **kwargs):
        self.output_dim = output_dim
        self.name = "sigma_"+name_suffix
        super(ConcatenateOutputWithSigma, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(1,),
                                      name=self.name,
                                      initializer='zeros',
                                      trainable=True)
        super(ConcatenateOutputWithSigma, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_ones_matrix = ((K.abs(x)+1)/(K.abs(x)+1))
        batch_kernel = K.expand_dims(input_ones_matrix[:,0], -1)*self.kernel
        return K.concatenate((x, batch_kernel), -1)

    def compute_output_shape(self, input_shape):
        return self.output_dim


def ontology_init(weights):
    def my_init(shape, dtype=None):
        return K.variable(weights)
    return my_init

class OntologyLayer(Layer):

    def __init__(self, output_dim, ontology_fpath, trainable = False,**kwargs):
        self.output_dim = output_dim
        self.ontology_fpath = ontology_fpath
        self.trainable = trainable
        super(OntologyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[1],self.output_dim[1]),
                                      name="ontology_kernel",
                                      initializer=ontology_init(np.load(self.ontology_fpath)),
                                      trainable=self.trainable)
        #self.kernel = K.variable(np.load(self.ontology_fpath))
        super(OntologyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return self.output_dim

    def get_config(self):
        #base_config = super(OntologyLayer, self).get_config()
        #base_config['output_dim'] = self.output_dim
        #base_config['ontology_fpath'] = self.ontology_fpath
        #base_config['trainable'] = self.trainable
        config = {
            'output_dim': (None,525),
            'ontology_fpath' : 'datasets/VireoFood172/meta/ontology_files/DI_ontology_matrix.npy',
            'trainable' : False}
        print(config)
        base_config = super(OntologyLayer, self).get_config()
        #return dict(list(base_config.items()) + list(config.items()))
        #return base_config
        return config

class OntologyLayerV2(Layer):

    def __init__(self, output_dim, ontology_fpath, **kwargs):
        self.output_dim = output_dim
        self.ontology_fpath = ontology_fpath
        super(OntologyLayerV2, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(shape=(input_shape[1],self.output_dim[1]),
                                      name="ontology_kernel",
                                      initializer=ontology_init(np.load(self.ontology_fpath)),
                                      trainable=False)
        #self.kernel = K.variable(np.load(self.ontology_fpath))
        super(OntologyLayerV2, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # (bs,nconcepts)
        out = K.expand_dims(x, axis=-1)
        # (bs,nconcepts, 1)
        out = out * self.kernel
        # (bs,nconcepts,nconcepts)
        
        out_wo_diag = out*(1-K.variable(np.identity(self.output_dim[1])))
        min_vals = K.min(out_wo_diag, axis=1)
        noutput = min_vals+x
        #max_vals = K.max(out_wo_diag, axis=1)
        #noutput = min_vals+max_vals+x
        
        return noutput
        #return K.sum(out, axis = 1)
        #return K.dot(x, self.kernel)
        #return x
    def compute_output_shape(self, input_shape):
        return self.output_dim
