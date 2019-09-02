import numpy as np
import pandas as pd


class FoodOntology:

    def __init__(self, params):

        self.concepts_file = params['concepts_list']
        self.dish_list_file = params['dish_list']
        self.dish_ingr_prob_file = params['dish_ingr_prob_file']
        self.ingr_dish_prob_file = params['ingr_dish_prob_file']
        self.ingr_ingr_prob_file = params['ingr_ingr_prob_file']
        self.store_path = params['store_path']
        self.probabilities = params['probabilities']
        self.ont_type = params['ontology_type'] # (0,1),(-1,1),(probs)

        with open(self.concepts_file) as f:

            self.concept_order = f.readlines()
            # Read the concepts order and bulk it into a list
            for i, concept in enumerate(self.concept_order):
                self.concept_order[i] = concept.replace('\n', '')

        with open(self.dish_list_file) as f:

            dish_list = []
            for line in f:
                dish_list.append(line.replace('\n', ''))

        self.n_dishes = len(dish_list)

        self.dish_order = self.concept_order[:self.n_dishes]
        self.ingr_order = self.concept_order[self.n_dishes:]

        self.dishes_dict = {concept: i for i, concept in enumerate(self.dish_order)}
        self.ingredients_dict = {concept: i for i, concept in enumerate(self.ingr_order)}

        self.n_ingredients = len(self.ingredients_dict)

        self.dish_to_idx_dict = {value: key for (key, value) in self.dishes_dict.items()}
        self.ingredients_to_idx_dict = {value: key for (key, value) in self.ingredients_dict.items()}

    def build_ontology(self):

        # Load TopDown Probabilities Dish-Ingredient
        dish_ingr_df = pd.read_csv(self.dish_ingr_prob_file, header=0,
                                   names=['Dish', 'Ingredient', 'Ingredient Probability'])
        dish_ingr_df.Dish = dish_ingr_df.Dish.apply(lambda x: x.replace('((', ''))
        dish_ingr_df.Dish = dish_ingr_df.Dish.apply(lambda x: x.replace('\'', ''))
        dish_ingr_df['Ingredient'] = dish_ingr_df['Ingredient'].apply(lambda x: x.replace(')', ''))
        dish_ingr_df['Ingredient'] = dish_ingr_df['Ingredient'].apply(lambda x: x.replace('\'', ''))
        dish_ingr_df['Ingredient'] = dish_ingr_df['Ingredient'].apply(lambda x: x.replace(' ', ''))
        dish_ingr_df['Ingredient Probability'] = dish_ingr_df['Ingredient Probability'].apply(
            lambda x: x.replace(')', ''))
        dish_ingr_df['Ingredient Probability'] = dish_ingr_df['Ingredient Probability'].apply(
            lambda x: float(x.replace(' ', '')))

        #Load Bottom-Up Ingredient-Dish Probabilities

        ingr_dish_df = pd.read_csv(self.ingr_dish_prob_file, header=0, names=['Ingredient','Dish', 'Ingr-Dish Prob'])

        # Load Coexistence Probabilities Ingredient-Ingredient
        ingr_pair_df = pd.read_csv(self.ingr_ingr_prob_file, index_col=False, header=0,
                                   names=['Dish', 'Ingr1', 'Ingr2', 'Coexistence Probability'])
        ingr_pair_df.Dish = ingr_pair_df.Dish.apply(lambda x: x.replace('((', ''))
        ingr_pair_df.Dish = ingr_pair_df.Dish.apply(lambda x: x.replace('\'', ''))
        ingr_pair_df.Ingr1 = ingr_pair_df.Ingr1.apply(lambda x: x.replace('(', ''))
        ingr_pair_df.Ingr1 = ingr_pair_df.Ingr1.apply(lambda x: x.replace('\'', '').replace('"', '').replace(' ', ''))
        ingr_pair_df.Ingr2 = ingr_pair_df.Ingr2.apply(lambda x: x.replace(')")', ''))
        ingr_pair_df.Ingr2 = ingr_pair_df.Ingr2.apply(lambda x: x.replace('\'', '').replace(' ', ''))
        ingr_pair_df['Coexistence Probability'] = ingr_pair_df['Coexistence Probability'].apply(
            lambda x: x.replace(')', ''))
        ingr_pair_df['Coexistence Probability'] = ingr_pair_df['Coexistence Probability'].apply(
            lambda x: float(x.replace(' ', '')))

        coex_prob_df = ingr_pair_df[['Ingr1', 'Ingr2']].merge(
            ingr_pair_df[['Ingr1', 'Ingr2', 'Coexistence Probability']].groupby(['Ingr1', 'Ingr2']).sum(),
            left_on=['Ingr1', 'Ingr2'], right_index=True).drop_duplicates()

        # Build submatrix Dish-Dish (:n_dishes,:n_dishes)
        dish_dish_matrix = np.eye(self.n_dishes)

        if self.ont_type == -1:

            dish_dish_matrix[np.where(dish_dish_matrix == 0)] = -1

        # Build submatrix Dish-Ingredient (:n_dishes,n_dishes:)
        dish_ingredient_matrix = np.zeros([self.n_dishes, self.n_ingredients])

        # 0 -> -1
        #dish_ingredient_matrix = -1*np.ones([self.n_dishes, self.n_ingredients])

        if self.probabilities:

            for value1, key1 in self.dishes_dict.items():
                for value2, key2 in self.ingredients_dict.items():

                    if value1 != value2 and value1 in self.dish_order:

                        if not dish_ingr_df[(dish_ingr_df.Dish == value1) & (dish_ingr_df.Ingredient == value2)][
                            'Ingredient Probability'].empty:

                            try:

                                dish_ingredient_matrix[key1, key2] = \
                                    dish_ingr_df[(dish_ingr_df.Dish == value1) & (dish_ingr_df.Ingredient == value2)][
                                        'Ingredient Probability']

                            except:

                                dish_ingredient_matrix[key1, key2] = 0

                        else:

                            dish_ingredient_matrix[key1, key2] = 0

        else:

            for value1, key1 in self.dishes_dict.items():
                for value2, key2 in self.ingredients_dict.items():

                    if value1 != value2 and value1 in self.dish_order:

                        if not dish_ingr_df[(dish_ingr_df.Dish == value1) & (dish_ingr_df.Ingredient == value2)][
                            'Ingredient Probability'].empty:

                            try:

                                dish_ingredient_matrix[key1, key2] = 1

                            except:

                                if self.ont_type == 0:

                                    dish_ingredient_matrix[key1, key2] = 0

                                elif self.ont_type == -1:

                                    dish_ingredient_matrix[key1, key2] = -1

                        else:

                            if self.ont_type == 0:

                                dish_ingredient_matrix[key1, key2] = 0

                            elif self.ont_type == -1:

                                dish_ingredient_matrix[key1, key2] = -1

                        # save TopDown Ontology into an npy file
        #np.save(self.store_path + 'TopDown_Ontology_matrix', dish_ingredient_matrix)

        # Build submatrix Ingredient-Dish (n_dishes:,:n_dishes)

        ingredient_dish_matrix = np.zeros((self.n_ingredients, self.n_dishes))

        if self.probabilities:

            for value1, key1 in self.dishes_dict.items():
                for value2, key2 in self.ingredients_dict.items():

                    if value1 != value2 and value1 in self.dish_order:

                        if not ingr_dish_df[(ingr_dish_df.Dish == value1) & (ingr_dish_df.Ingredient == value2)][
                            'Ingr-Dish Prob'].empty:

                            try:

                                ingredient_dish_matrix[key2, key1] = \
                                    ingr_dish_df[(ingr_dish_df.Dish == value1) & (ingr_dish_df.Ingredient == value2)][
                                        'Ingr-Dish Prob']

                            except:

                                ingredient_dish_matrix[key2, key1] = 0

                        else:

                            ingredient_dish_matrix[key2, key1] = 0

        else:

            for value1, key1 in self.dishes_dict.items():
                for value2, key2 in self.ingredients_dict.items():

                    if value1 != value2 and value1 in self.dish_order:

                        if not ingr_dish_df[(ingr_dish_df.Dish == value1) & (ingr_dish_df.Ingredient == value2)][
                            'Ingr-Dish Prob'].empty:

                            try:

                                ingredient_dish_matrix[key2, key1] = 1

                            except:

                                if self.ont_type == 0:

                                    ingredient_dish_matrix[key2, key1] = 0

                                elif self.ont_type == -1:

                                    ingredient_dish_matrix[key2, key1] = -1

                        else:

                            if self.ont_type == 0:

                                ingredient_dish_matrix[key2, key1] = 0

                            elif self.ont_type == -1:

                                ingredient_dish_matrix[key2, key1] = -1

        # Build submatrix Ingredient-Ingredient (n_dishes:, n_dishes:)

        ingredient_ingredient_matrix = np.eye(self.n_ingredients)

        if self.probabilities:

            for value1, key1 in self.ingredients_dict.items():
                for value2, key2 in self.ingredients_dict.items():

                    if value1 != value2:

                        if not coex_prob_df[(coex_prob_df.Ingr1 == value1) & (coex_prob_df.Ingr2 == value2)][
                            'Coexistence Probability'].empty:

                            ingredient_ingredient_matrix[key1, key2] = \
                                coex_prob_df[(coex_prob_df.Ingr1 == value1) & (coex_prob_df.Ingr2 == value2)][
                                    'Coexistence Probability']
                            ingredient_ingredient_matrix[key2, key1] = \
                                coex_prob_df[(coex_prob_df.Ingr1 == value1) & (coex_prob_df.Ingr2 == value2)][
                                    'Coexistence Probability']

                        else:

                            ingredient_ingredient_matrix[key1, key2] = 0
                            ingredient_ingredient_matrix[key2, key1] = 0

        else:

            for value1, key1 in self.ingredients_dict.items():
                for value2, key2 in self.ingredients_dict.items():

                    if value1 != value2:

                        if not coex_prob_df[(coex_prob_df.Ingr1 == value1) & (coex_prob_df.Ingr2 == value2)][
                            'Coexistence Probability'].empty:

                            ingredient_ingredient_matrix[key1, key2] = 1
                            ingredient_ingredient_matrix[key2, key1] = 1

                        else:

                            if self.ont_type == 0:

                                ingredient_ingredient_matrix[key1, key2] = 0
                                ingredient_ingredient_matrix[key2, key1] = 0

                            elif self.ont_type == -1:

                                ingredient_ingredient_matrix[key1, key2] = -1
                                ingredient_ingredient_matrix[key2, key1] = -1

            # Join all together
        di_ontology_matrix = np.block([[dish_dish_matrix, dish_ingredient_matrix],
                                    [np.zeros([self.n_ingredients, self.n_dishes]), np.eye(self.n_ingredients)]])

        id_ontology_matrix = np.block([[dish_dish_matrix, np.zeros([self.n_dishes,self.n_ingredients])],
                                    [ingredient_dish_matrix, np.eye(self.n_ingredients)]])

        di_id_ontology_matrix = np.block([[dish_dish_matrix, dish_ingredient_matrix],
                                              [ingredient_dish_matrix, np.eye(self.n_ingredients)]])

        ii_ontology_matrix = np.block([[dish_dish_matrix, np.zeros([self.n_dishes, self.n_ingredients])],
                                          [np.zeros([self.n_ingredients, self.n_dishes]), ingredient_ingredient_matrix]])

        full_ontology_matrix = np.block([[dish_dish_matrix, dish_ingredient_matrix],
                                          [ingredient_dish_matrix, ingredient_ingredient_matrix]])



        # save Global Graph Ontology into an npy file

        if self.probabilities:

            np.save(self.store_path + 'Ontologies_probabilities/' + 'DI_ontology_matrix', di_ontology_matrix)
            np.save(self.store_path + 'Ontologies_probabilities/' + 'ID_ontology_matrix', id_ontology_matrix)
            np.save(self.store_path + 'Ontologies_probabilities/' + 'DI_ID_ontology_matrix', di_id_ontology_matrix)
            np.save(self.store_path + 'Ontologies_probabilities/' + 'II_ontology_matrix', ii_ontology_matrix)
            np.save(self.store_path + 'Ontologies_probabilities/' + 'FULL_ontology_matrix', full_ontology_matrix)

        else:

            if self.ont_type == 0:

                np.save(self.store_path + 'Ontologies_1_0/' + 'DI_ontology_matrix', di_ontology_matrix)
                np.save(self.store_path + 'Ontologies_1_0/' + 'ID_ontology_matrix', id_ontology_matrix)
                np.save(self.store_path + 'Ontologies_1_0/' + 'DI_ID_ontology_matrix', di_id_ontology_matrix)
                np.save(self.store_path + 'Ontologies_1_0/' + 'II_ontology_matrix', ii_ontology_matrix)
                np.save(self.store_path + 'Ontologies_1_0/' + 'FULL_ontology_matrix', full_ontology_matrix)

            elif self.ont_type == -1:

                np.save(self.store_path + 'Ontologies_1_-1/' + 'DI_ontology_matrix', di_ontology_matrix)
                np.save(self.store_path + 'Ontologies_1_-1/' + 'ID_ontology_matrix', id_ontology_matrix)
                np.save(self.store_path + 'Ontologies_1_-1/' + 'DI_ID_ontology_matrix', di_id_ontology_matrix)
                np.save(self.store_path + 'Ontologies_1_-1/' + 'II_ontology_matrix', ii_ontology_matrix)
                np.save(self.store_path + 'Ontologies_1_-1/' + 'FULL_ontology_matrix', full_ontology_matrix)

    def return_concepts(self):

        return self.concept_order

    def info(self):

        print('Number of dishes:', self.n_dishes)
        print('Number of ingredients:', self.n_ingredients)


def main(params):

    ont = FoodOntology(params)
    ont.build_ontology()


if __name__ == '__main__':
    params = {
      'concepts_list': '../datasets/Recipes5k/mtannotations/concepts_list.txt',
      'dish_list': '../datasets/Recipes5k/mtannotations/dish_list.txt',
      'dish_ingr_prob_file': '../datasets/Recipes5k/mtannotations/dish_ingr_probabilities.txt',
      'ingr_dish_prob_file': '../datasets/Recipes5k/mtannotations/ingr_dish_probabilities.txt',
      'ingr_ingr_prob_file': '../datasets/Recipes5k/mtannotations/dish_ingr_pair_coexistence_probability.txt',
      'store_path': '../datasets/Recipes5k/mtannotations/',
      'probabilities': True, #True or False
      'ontology_type': 0 #0 or -1
      }
    main(params)

