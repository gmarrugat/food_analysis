import numpy as np

def build_ontologies(full_ontology_file, n_dishes, n_ingr, store_path):
    
    full_ont = np.load(full_ontology_file)
    n_elem = n_dishes+n_ingr
    
    di_aux_matrix = np.block([[np.ones([n_dishes,n_dishes]), np.ones([n_dishes, n_ingr])],
                          [np.zeros([n_ingr, n_dishes]), np.zeros([n_ingr, n_ingr])]])

    id_aux_matrix = np.block([[np.ones([n_dishes,n_dishes]), np.zeros([n_dishes, n_ingr])],
                          [np.ones([n_ingr, n_dishes]), np.zeros([n_ingr, n_ingr])]])

    di_id_aux_matrix = np.block([[np.ones([n_dishes,n_dishes]), np.ones([n_dishes, n_ingr])],
                          [np.ones([n_ingr, n_dishes]), np.zeros([n_ingr, n_ingr])]])

    ii_aux_matrix = np.block([[np.ones([n_dishes,n_dishes]), np.zeros([n_dishes, n_ingr])],
                          [np.zeros([n_ingr, n_dishes]), np.ones([n_ingr, n_ingr])]])
    
    #Dish-Ingredient
    di_ontology_matrix = (full_ont - np.diag(full_ont.diagonal()))*di_aux_matrix + np.eye(n_elem)
    #Ingredient-Dish
    id_ontology_matrix = (full_ont - np.diag(full_ont.diagonal()))*id_aux_matrix + np.eye(n_elem)
    #Dish-Ingredient_Ingredient-Dish
    di_id_ontology_matrix = (full_ont - np.diag(full_ont.diagonal()))*di_id_aux_matrix + np.eye(n_elem)
    #Ingredient-Ingredient
    ii_ontology_matrix = (full_ont - np.diag(full_ont.diagonal()))*ii_aux_matrix + np.eye(n_elem)
    #Dish-Ingredient Ingredient-Ingredient
    di_ii_ontology_matrix = di_ontology_matrix + ii_ontology_matrix
    
    np.save(store_path+'DI_ontology_matrix',di_ontology_matrix)
    np.save(store_path+'ID_ontology_matrix',id_ontology_matrix)
    np.save(store_path+'DI_ID_ontology_matrix',di_id_ontology_matrix)
    np.save(store_path+'II_ontology_matrix',ii_ontology_matrix)
    np.save(store_path+'DI_II_ontology_matrix',di_ii_ontology_matrix)
    np.save(store_path+'FULL_ontology_matrix',full_ont)

	
ds_root = "../datasets/VireoFood172/meta/"
d_list_fname = ds_root+"FoodList_2.txt"
ing_list_fname = ds_root+"IngredientList_2.txt"

d_classes = []

concepts_fnames = [d_list_fname, ing_list_fname]
concepts_prefix = ["d_", "ing_"] # some dishes and ingredients may have the same name.

concepts = []
for i in range(len(concepts_fnames)):
    with open(concepts_fnames[i], "r") as f:
        for line in f:
            concepts.append(concepts_prefix[i]+line.strip())
            if i ==0:
                d_classes.append(line.strip())

##print len(concepts), len(set(concepts))

ontology_matrix = np.zeros((len(concepts), len(concepts)))
recipesxconcept = np.zeros((len(concepts), 1))

train_fname = ds_root+"train.txt"
with open(train_fname, "r") as f:
    train = [line.strip() for line in f]

train_d_lbls_fname = ds_root+"train_dish_lbls.txt"
with open(train_d_lbls_fname, "r") as f:
    train_d_lbls = [line.strip() for line in f]

train_ing_lbls_fname = ds_root+"train_ing_lbls_2.txt"
with open(train_ing_lbls_fname, "r") as f:
    train_ing_lbls = [line.strip() for line in f]

ing_classes_fname = ds_root+"ingredients.txt"
with open(ing_classes_fname, "r") as f:
    ing_classes = [line.strip() for line in f]


for d_lbl, ing_lbls in zip(train_d_lbls, train_ing_lbls):
    d_ont_idx = concepts.index("d_"+d_classes[int(d_lbl)])
    ing_ont_idx_l = [concepts.index("ing_"+cls) for cls in ing_classes[int(ing_lbls)].split(",")]
    recipesxconcept[d_ont_idx][0] = recipesxconcept[d_ont_idx][0] + 1
    for i in range(len(ing_ont_idx_l)):
        ing_ont_idx = ing_ont_idx_l[i]
        recipesxconcept[ing_ont_idx][0] = recipesxconcept[ing_ont_idx][0] + 1
        # top down
        ontology_matrix[d_ont_idx, ing_ont_idx] = ontology_matrix[d_ont_idx, ing_ont_idx] + 1
        # bottom up
        ontology_matrix[ing_ont_idx, d_ont_idx] = ontology_matrix[ing_ont_idx, d_ont_idx] + 1

    # coexistence
    for i in range(len(ing_ont_idx_l)-1):
        for j in range(i+1, len(ing_ont_idx_l)):
            ing_ont_idx_i = ing_ont_idx_l[i]
            ing_ont_idx_j = ing_ont_idx_l[j]
            # left to right
            ontology_matrix[ing_ont_idx_i, ing_ont_idx_j] = ontology_matrix[ing_ont_idx_i, ing_ont_idx_j] + 1
            # right to left 
            ontology_matrix[ing_ont_idx_j, ing_ont_idx_i] = ontology_matrix[ing_ont_idx_j, ing_ont_idx_i] + 1

#print np.sum(ontology_matrix)
#print ontology_matrix[0]
#print recipesxconcept[0]

np.save(ds_root+"ontology_files/ontology_matrix_freq.npy", ontology_matrix)
np.save(ds_root+"ontology_files/recipes_x_concept.npy", recipesxconcept)

ontology_matrix_1 = np.copy(ontology_matrix) + np.identity(len(concepts))
for i in range(len(concepts)):
    for j in range(len(concepts)):
        if ontology_matrix_1[i][j]>0:
            ontology_matrix_1[i][j] = 1

np.save(ds_root+"ontology_files/ontology_matrix_1.npy", ontology_matrix_1)
#print ontology_matrix_1[0]
ontology_matrix_1_neg = np.copy(ontology_matrix_1)
for i in range(len(concepts)):
    for j in range(len(concepts)):
        if ((j >= len(d_classes) and i<len(d_classes)) or i>=len(d_classes)) and ontology_matrix_1_neg[i][j]==0:
            ontology_matrix_1_neg[i][j] = -1

np.save(ds_root+"ontology_files/ontology_matrix_1_neg.npy", ontology_matrix_1_neg)
#print ontology_matrix_1_neg[0]

ontology_matrix_prob = ontology_matrix / recipesxconcept + np.identity(len(concepts))
np.save(ds_root+"ontology_files/ontology_matrix_prob.npy", ontology_matrix_prob)
#print ontology_matrix_prob[0]
ontology_matrix_prob_neg = np.copy(ontology_matrix_prob)
for i in range(len(concepts)):
    for j in range(len(concepts)):
        if ((j >= len(d_classes) and i<len(d_classes)) or i>=len(d_classes)) and ontology_matrix_prob_neg[i][j]==0:
            ontology_matrix_prob_neg[i][j] = -1*(1/float(recipesxconcept[i][0]))

#print ontology_matrix_prob_neg[0]

np.save(ds_root+"ontology_files/ontology_matrix_prob_neg.npy", ontology_matrix_prob_neg)

ont_1_file = ds_root+'ontology_files/ontology_matrix_1.npy'
ont_1_neg_file = ds_root+'ontology_files/ontology_matrix_1_neg.npy'
ont_probs_file = ds_root+'ontology_files/ontology_matrix_prob.npy'
ont_prob_neg_file = ds_root+'ontology_files/ontology_matrix_prob_neg.npy'

n_dishes = len(d_classes)
n_ingr = len(concepts) - len(d_classes)

build_ontologies(ont_1_file, n_dishes, n_ingr, ds_root+'ontology_files/Ontologies_1/')
build_ontologies(ont_1_neg_file, n_dishes, n_ingr, ds_root+'ontology_files/Ontologies_1_neg/')
build_ontologies(ont_probs_file, n_dishes, n_ingr, ds_root+'ontology_files/Ontologies_prob/')
build_ontologies(ont_prob_neg_file, n_dishes, n_ingr, ds_root+'ontology_files/Ontologies_prob_neg/')

