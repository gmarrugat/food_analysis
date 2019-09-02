# Applying Deep Learning for Food Image Analysis

# Abstract

Food is an important component in people's daily life, examples of the
previous assertion are the range of possible diets according to the animal or
vegetal origin, the intolerance to some aliments and lately the increasing number
of food pictures in social networks. Several computer vision approaches have
been proposed for tackling food analysis problems, but few effort has been done
in taking benefit of the hierarchical relation between elements in a food image;
dish and ingredients.
In this project the highly performing state of the art CNN method is
adapted concatenating an ontology layer, a multidimensional layer which contains
the relation between the elements, in order to help during the classification
process. Different structures for the ontology have been tested to prove which
relations have the most beneficial impact, and which are less relevant. Additionally
to structure, the value of the elements that compound this hierarchical
relation layer play an important role, therefore the experiments performed contained
different weighted relations between the components. The ontology layer
is built with the labels of the multiple task in the dataset used to train the model.
At the end, the results obtained will be compared to a baseline model without
the ontology layer and it will be appreciated how hierarchical relations between
tasks benefits classification. Finally, the result will be a model which will be
able to simultaneously predict two food-related tasks; dish and ingredients.


# Prepare environment

To be able to run the code in this repository it is required to create an environment with the following packages:

* tensorflow
* keras
* numpy
* pandas
* matplotlib
* scikit-learn
* multimodal-keras-wrapper

If the code is desired to be run on a GPU (install tensorflow-gpu), it is important to take care about the compatibility of the CUDA version with packages versions. In our case, it was CUDA 8.0, compatible with tensorflow-gpu == 1.4.0 and keras == 2.2, which force us to create a python 2.7 environment.

# Prepare Dataset

Download the [Recipes5k](http://www.ub.edu/cvub/recipes5k/) and [Vireo-Food 172](http://vireo.cs.cityu.edu.hk/VireoFood172/) datasets and locate them in dataset directory in their own respective folders. Each download file contains the whole image database, class labels files and train, validation and test split files. 

# Building Ontology
 
The  methodology to build the ontology is different for the two datasets.

For Recipes5k, the notebook Recipes5k_prepare_data creates the files which contain the different relations, probabilities and concepts list. Once these files are generated, executing Food_Analysis.py file will generate the different Ontology files structures. At the end of the script Food_Analysis.py there is the execution parameters, probabilities and ontology_type define which type of relational value will conform the ontology generated.

It was the beginning of the project and we were playing around with the data, that´s why we used a notebook. 

For VireoFood-172, just run make_ontology_matrix.py and all the different structure value combinations will be built.
