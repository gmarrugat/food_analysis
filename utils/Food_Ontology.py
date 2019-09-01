import numpy as np
import pandas as pd

class Food_Ontology():
  
  def __init__(self, params):
    
    self.concepts_file = params['concepts_list']
    self.dish_list_file = params['dish_list']
    self.dish_ingr_prob_file = params['dish_ingr_prob_file']
    self.ingr_ingr_prob_file = params['ingr_ingr_prob_file']
    self.store_path = params['store_path']
    self.probabilities = params['probabilities']
    
    with open(self.concepts_file) as f:

      concept_order = f.readlines()
      #Read the concepts order and bulk it into a list
      for i,concept in enumerate(concept_order):

        concept_order[i] = concept.replace('\n','')
        
    with open(self.dish_list_file) as f:
  
      dish_list = []
      for line in f:
        dish_list.append(line.replace('\n',''))
    
    self.n_dishes = len(dish_list)
        
    self.dish_order = concept_order[:n_dishes]
    self.ingr_order = concept_order[n_dishes:]
    
    self.dishes_dict = {concept:i for i,concept in enumerate(self.dish_order)}
    self.ingredients_dict = {concept:i for i,concept in enumerate(self.ingr_order)}
    
    self.n_ingredients = len(self.ingredients_dict)

    self.dish_to_idx_dict = {value:key for (key,value) in self.dishes_dict.items()}
    self.ingredients_to_idx_dict = {value:key for (key,value) in self.ingredients_dict.items()}
    
  def build_Ontology(self):
    
    #Load TopDown Probabilities Dish-Ingredient
    dish_ingr_df = pd.read_csv(self.dish_ingr_prob_file, header=0 , names=['Dish','Ingredient','Ingredient Probability'])
    dish_ingr_df.Dish = dish_ingr_df.Dish.apply(lambda x: x.replace('((',''))
    dish_ingr_df.Dish= dish_ingr_df.Dish.apply(lambda x: x.replace('\'',''))
    dish_ingr_df['Ingredient'] = dish_ingr_df['Ingredient'].apply(lambda x: x.replace(')',''))
    dish_ingr_df['Ingredient'] = dish_ingr_df['Ingredient'].apply(lambda x: x.replace('\'',''))
    dish_ingr_df['Ingredient'] = dish_ingr_df['Ingredient'].apply(lambda x: x.replace(' ',''))
    dish_ingr_df['Ingredient Probability'] = dish_ingr_df['Ingredient Probability'].apply(lambda x: x.replace(')',''))
    dish_ingr_df['Ingredient Probability'] = dish_ingr_df['Ingredient Probability'].apply(lambda x: float(x.replace(' ','')))
    
    #Load Coexistence Probabilities Ingredient-Ingredient
    ingr_pair_df = pd.read_csv(self.ingr_ingr_prob_file, index_col=False, header = 0 ,names=['Dish','Ingr1','Ingr2','Coexistence Probability'])
    ingr_pair_df.Dish = ingr_pair_df.Dish.apply(lambda x: x.replace('((',''))
    ingr_pair_df.Dish = ingr_pair_df.Dish.apply(lambda x: x.replace('\'',''))
    ingr_pair_df.Ingr1 = ingr_pair_df.Ingr1.apply(lambda x: x.replace('(',''))
    ingr_pair_df.Ingr1 = ingr_pair_df.Ingr1.apply(lambda x: x.replace('\'','').replace('"','').replace(' ',''))
    ingr_pair_df.Ingr2 = ingr_pair_df.Ingr2.apply(lambda x: x.replace(')")',''))
    ingr_pair_df.Ingr2 = ingr_pair_df.Ingr2.apply(lambda x: x.replace('\'','').replace(' ',''))
    ingr_pair_df['Coexistence Probability'] = ingr_pair_df['Coexistence Probability'].apply(lambda x: x.replace(')',''))
    ingr_pair_df['Coexistence Probability'] = ingr_pair_df['Coexistence Probability'].apply(lambda x: float(x.replace(' ','')))
    
    coex_prob_df = ingr_pair_df[['Ingr1','Ingr2']].merge(ingr_pair_df[['Ingr1','Ingr2','Coexistence Probability']].groupby(['Ingr1','Ingr2']).sum(), left_on=['Ingr1','Ingr2'], right_index = True).drop_duplicates()
    
    #Build submatrix Dish-Dish (:n_dishes,:n_dishes)
    Dish_Dish_Matrix = np.eye(self.n_dishes)
    
    #Build submatrix Dish-Ingredient (:n_dishes,n_dishes:)
    Dish_Ingredient_Matrix = np.zeros([self.n_dishes, self.n_ingredients])

    for value1,key1 in self.dishes_dict.items():
      for value2,key2 in self.ingredients_dict.items():
    
        if value1 != value2 and value1 in self.dish_order:
      
          if not dish_ingr_df[(dish_ingr_df.Dish == value1)&(dish_ingr_df.Ingredient == value2)]['Ingredient Probability'].empty:
        
            try:
            
              if self.probabilities == True:
                
                Dish_Ingredient_Matrix[key1,key2] = dish_ingr_df[(dish_ingr_df.Dish == value1)&(dish_ingr_df.Ingredient == value2)]['Ingredient Probability']
        
              else:
        
                Dish_Ingredient_Matrix[key1,key2] = 1
        
            except:
          
              Dish_Ingredient_Matrix[key1,key2] = 0
      
          else:
      
            pass 
      
      #save TopDown Ontology into an npy file
      np.save(self.store_path+'TopDown_Ontology_matrix',Dish_Ingredient_Matrix)
      
      #Build submatrix Ingredient-Dish (n_dishes:,:n_dishes) 
              #No BottomUp Ontology by the moment!!!!
      Ingredient_Dish_Matrix = np.zeros((self.n_ingredients,self.n_dishes))
      
      #Build submatrix Ingredient-Ingredient (n_dishes:, n_dishes:)
      
      if self.probabilities == True:
        
        Ingredient_Ingredient_Matrix = np.zeros([n_ingredients,n_ingredients])

        for value1, key1 in ingredients_dict.items():
          for value2, key2 in ingredients_dict.items():
      
            if not coex_prob_df[(coex_prob_df.Ingr1 == value1)&(coex_prob_df.Ingr2 == value2)]['Coexistence Probability'].empty:
        
              Ingredient_Ingredient_Matrix[key1,key2] = coex_prob_df[(coex_prob_df.Ingr1 == value1)&(coex_prob_df.Ingr2 == value2)]['Coexistence Probability']
              Ingredient_Ingredient_Matrix[key2,key1] = coex_prob_df[(coex_prob_df.Ingr1 == value1)&(coex_prob_df.Ingr2 == value2)]['Coexistence Probability']
      
            else:
      
              pass    
    
      else:
        
        Ingredient_Ingredient_Matrix = np.eye(self.n_ingredients)
        
      #Join all together
      Ontology_matrix = np.block([[Dish_Dish_Matrix,Dish_Ingredient_Matrix],
                                 [Ingredient_Dish_Matrix,Ingredient_Ingredient_Matrix]])
      
      #save Global Graph Ontology into an npy file
      np.save(self.store_path+'Ontology_matrix_v1',Ontology_matrix)
    
  def return_concepts(self):
    
    return concept_order
  
  def info(self):
    
    print('Number of dishes:', self.n_dishes)
    print('Number of ingredients:', self.n_ingredients)