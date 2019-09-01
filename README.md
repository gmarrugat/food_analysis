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
