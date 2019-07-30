'''
Skeleton Many to Many Linkage Code
For this code, deduplicating identical record pairs requires each...
... individual record to have a unique ID
 
Note: Lines 239 - 473 are the Expectation Maximisation Algorithm
'''

# Import any packages required
import pandas as pd
import numpy as np
from nltk.metrics import edit_distance
import time

# Time code
start = time.time()

# Folder Location for File
file_path = '__/__/__/'

# File Names of Datasets to be linked
csv_path_1 = file_path + '___.csv'
csv_path_2 = file_path + '___.csv'

# Required columns of dataset 1 e.g. 'forename_1'
columns_1 = ['___',
             '___',
             '___',
             '___',
             '___',
             '___']
             
# Required columns of dataset 2 e.g. 'forename_2'
columns_2 = ['___',
             '___',
             '___',
             '___',
             '___',
             '___']

# Types for dataset 1 e.g. 'forename_1' : 'str'
data_types_1 = {'___' : '___',
                '___' : '___',
                '___' : '___',
                '___' : '___',
                '___' : '___',
                '___' : '___'}
               
# Types for dataset 2 e.g. 'forename_2' : 'str'
data_types_2 = {'___' : '___',
                '___' : '___',
                '___' : '___',
                '___' : '___',
                '___' : '___',
                '___' : '___'}

# Read in the "rl1" csv
rl1 = pd.read_csv(csv_path_1, 
                  index_col=False, 
                  na_filter=False, 
                  encoding='ISO-8859-1',
                  dtype=data_types_1, 
                  usecols = columns_1)
                  
print("rl1 read in") 

# Read in the "rl2" csv
rl2 = pd.read_csv(csv_path_2, 
                  index_col=False, 
                  na_filter=False, 
                  encoding='ISO-8859-1',
                  dtype=data_types_2, 
                  usecols = columns_2)  
                   
print("rl2 read in")
 
# ------------------------------------------------------------------------ #
# ------------------------------ Blocking -------------------------------- #
# ------------------------------------------------------------------------ #    

'''

Example:

blocked_pairs = pd.merge(left=rl1,
                         right=rl2,
                         how="inner",
                         left_on=['forename_trigram_1', 'surname_trigram_1', 'day_of_birth_1'],
                         right_on=['forename_trigram_2', 'surname_trigram_2', 'day_of_birth_2']
                         )
                          
'''

blocked_pairs_1 = pd.merge(left=rl1,
                          right=rl2,
                          how="inner",
                          left_on =['___', '___', '___', '___', '___'],
                          right_on=['___', '___', '___', '___', '___']
                          )
                  
blocked_pairs_2 = pd.merge(left=rl1,
                          right=rl2,
                          how="inner",
                          left_on =['___', '___', '___', '___', '___'],
                          right_on=['___', '___', '___', '___', '___']
                          )
          
blocked_pairs_3 = pd.merge(left=rl1,
                          right=rl2,
                          how="inner",
                          left_on =['___', '___', '___', '___', '___'],
                          right_on=['___', '___', '___', '___', '___']
                          )

                  
# Concatanate all dataframes together - this gives us our candidate record pairs
# Our candidate pairs are referred to as 'combined_blocks' from this point
combined_blocks = pd.concat([blocked_pairs_1,
                             blocked_pairs_2,
                             blocked_pairs_3,
                             ])
                             
print("blocking complete")

# Memory clearance
del rl1
del rl2
del blocked_pairs_1
del blocked_pairs_2
del blocked_pairs_3


# ------------------------------------------------------------------------ #
# --------------------------- Deduplication ------------------------------ #
# ------------------------------------------------------------------------ # 


'''Duplicates Case 1 (AB - AB)'''

# Deduplicate Candidate Record Pairs
combined_blocks.drop_duplicates(subset=combined_blocks.columns, 
                                keep='first', 
                                inplace=True)

# Sort by all values
combined_blocks.sort_values(by=list(combined_blocks.columns), 
                            axis=0, 
                            na_position='first', 
                            inplace=True)
# Reset Index                            
combined_blocks.reset_index(drop=True, inplace=True)

'''Duplicates Case 2  (AB - BA)'''

# Create copy of data frame
combined_blocks2 = combined_blocks.copy()

# Place the smaller unique ID on the left, larger on the right
combined_blocks2.ID_1, combined_blocks2.ID_2 = np.where(combined_blocks2.ID_1 > combined_blocks2.ID_2,
                                                       [combined_blocks2.ID_2, combined_blocks2.ID_1],
                                                       [combined_blocks2.ID_1, combined_blocks2.ID_2])

# Drop rows with duplicate IDs 
combined_blocks2 = combined_blocks2.drop_duplicates(subset = ['ID_1', 'ID_2'])
   
# Return the index of the rows remaining 
index_values = combined_blocks2.index.values.tolist()

# Select rows of original df using index_values
combined_blocks = combined_blocks.loc[index_values]

# Sort by column values & Reset Index
combined_blocks.sort_values(by=list(combined_blocks.columns), axis=0, na_position='first', inplace=True)
combined_blocks.reset_index(drop=True, inplace=True)  

'''Duplicates Case 3 (AA)'''

combined_blocks = combined_blocks[combined_blocks['ID_1'] != combined_blocks['ID_2']]
combined_blocks.reset_index(drop=True, inplace=True)    

print("combined_blocks successfully deduplicated")

# Memory Clearance
del combined_blocks2
del index_values


# ------------------------------------------------------------------- #
# ----------------- Calculate agreement vector pairs ---------------- #
# ------------------------------------------------------------------- #


'''Agreement vector is created which is then inputted into the EM Algorithm.
Set v1, v2, v3, v4, v5... as the agreement variables

Choose here what variables require partial agreement (edit distance - e.g. forename) and ...
... what variables do not (e.g. DOB, Postcode)'''

# Select agreement variables (e.g. forename, surname, DOB, Postcode, Country of Birth)
v1 = 'variable1'
v2 = 'variable2'
v3 = 'variable3'
v4 = 'variable4'
v5 = 'variable5'

# All agreement variables used to calculate match weights & probabilities
all_variables = [v1, v2, v3, v4, v5]

# Variables using partial agreement (string similarity)
edit_distance_variables = [v1, v2, v3]

# Cut off values for edit distance variables
# Example: Set agreement for pairs with forename edit distances below 0.35 to 0.00
# If no cutoff is required then remove variable from 'edit_distance_variables' and add to 'remaining_variables'
cutoff_values = [0.00, 0.75, 0.35]

# Remaining Variables - Only zero or full agreement for these
remaining_variables = [v4, v5]

# Agreement Vector Columns for each variable - used for EM
for variable in all_variables:
    combined_blocks[variable + '_agree'] = np.where(combined_blocks[variable + '_1'] == combined_blocks[variable + '_2'], 1., 0.)

# Save to New Dataframe - Agreement Matrix
agreement_matrix = combined_blocks[[v1 + '_agree', 
                                    v2 + '_agree',
                                    v3 + '_agree',
                                    v4 + '_agree',
                                    v5 + '_agree']]
print("Agreement vector created")                                          
                      
# ------------------------------------------------------------------- #
# ------------- B Expectation Maximization Algorithm  --------------- #
# ------------------------------------------------------------------- #

'''
EM - Algorithm Code    
Input: Agreement vector matrix + initial M/U values + initial prior
Output: - M/U values for each variable + prior
'''

# Arrays used for EM for loops
j_record_pairs = np.arange(0, agreement_matrix.values.shape[0])
i_variable_parameters = np.arange(0, agreement_matrix.values.shape[1])

# Number of Record Pairs
number_record_pairs = agreement_matrix.values.shape[0]

# Number of Variables 
number_variable_parameters = agreement_matrix.values.shape[1]

# Initial M and U values for each variable
m_parameter = np.full(number_variable_parameters, 0.9)
u_parameter = np.full(number_variable_parameters, 0.1)

# These aren't used in the EM 
# m_default = np.full(number_variable_parameters, 0.9)
# u_default = np.full(number_variable_parameters, 0.1)

# Inital prior: Proportion of all candidate record pairs that we think are true
prior_initial = 0.50

# Value required for convergence
deltamu_convergence = 0.00001

# Max no. of iterations allowed
max_iteration = 100

# ---------- Initialise Starting Variables -------------------------- #
#   Initialise Placeholder Variables for:
#   Gamma Matrix                -> gamma
#   Indicator Function          -> gj (For M Values) (Notes use a single gj but code has one for M and one for U)
#   Indicator Function          -> gj (For U Values) (Notes use a single gj but code has one for M and one for U)
#   Prior Estimate              -> p_hat
#   Initial (ith) m Estimate    -> m_i
#   Initial (ith) u Estimate    -> u_i
# ------------------------------------------------------------------- #

# Copy of agreement matrix
gamma = np.copy(agreement_matrix)

# Array of zeros (will get updated)
gj_m = np.zeros(agreement_matrix.values.shape[0])

# Array of zeros (will get updated)
gj_u = np.zeros(agreement_matrix.values.shape[0])

# Copy of inital prior value (will get updated)
prior = np.copy(prior_initial)

# Copy of intial m values (will get updated)
m_1 = np.copy(m_parameter)

# Copy of intial u values (will get updated)
u_1 = np.copy(u_parameter)

# Delta MU intial value - we want this to converge
delta_mu = 5.

# Start at iteration 0 
iteration_count = 0

# ------------------------------------------------------------------------------------- #
# ----------------- Start Main Loop - Run Until Stopping Criteria Met ----------------- #
# ------------------------------------------------------------------------------------- #

while (delta_mu > deltamu_convergence) & (iteration_count < max_iteration):
    
    # ------ Run E Step ------ #
    
    """
    Calculate the Indicator Variable gj_m and gj_u for all j record pairs (Eq 9.5 & 9.5*)
    """

    # Iterate over all record pairs
    for j in j_record_pairs:
        
        # Initialise Product for gm_u and gj_u
        p_g_agree_vector_match = 1.
        p_g_agree_vector_nonmatch = 1.

        # Iterate over all Variables
        for i in i_variable_parameters:
        
            p_g_agree_vector_match = (p_g_agree_vector_match * (m_1[i] ** gamma[j, i]) * (1 - m_1[i]) ** (1 - gamma[j, i]))
            p_g_agree_vector_nonmatch = (p_g_agree_vector_nonmatch * (u_1[i] ** gamma[j, i]) * (1 - u_1[i]) ** (1 - gamma[j, i]))

        # Calculate indicator functions gj_m and gj_u (9.5 and 9.5*)
        gj_m[j] = prior * p_g_agree_vector_match / (prior * p_g_agree_vector_match + (1 - prior) * p_g_agree_vector_nonmatch)
        gj_u[j] = (1 - prior) * p_g_agree_vector_nonmatch / (prior * p_g_agree_vector_match + (1 - prior) * p_g_agree_vector_nonmatch)

        # Error Check on indicator functions gj_m and gj_u (Check if Division by Zero)
        if gj_m[j] == 0 or gj_u[j] == 0:
            error = "gj_m[j] == 0 or gj_u[j] == 0"
            end = True
            break
        else:
            end = False

    if end:
        print(error)
        break

    # ------ End of E Step ------ #

    # ------ Run M Step --------- #
    
    """
    Calculate Estimates for m, u and p, for all record pairs (Eq 9.6, 9,7, 9.8)
    """
    
    # Record Original m_i and u_i
    previous_m = np.copy(m_1)
    previous_u = np.copy(u_1)

    # Initialise Denominator Sum (Eqn 9.6)
    # Start gj_m and gj_u sum at zero
    gj_m_sum = 0.
    gj_u_sum = 0.

    # Initialise Estimate for new m1 and u1, start sum at zero
    m_1 = np.zeros(agreement_matrix.values.shape[1])
    u_1 = np.zeros(agreement_matrix.values.shape[1])

    # Iterate over all j records
    for j in j_record_pairs:
        
        # Iterate over all i parameter variables
        for i in i_variable_parameters:
            
            # Update the m and u values for each variable
            m_1[i] = m_1[i] + gamma[j, i] * gj_m[j]
            u_1[i] = u_1[i] + gamma[j, i] * gj_u[j]

        # Update gj_m and gj_u sum:
        gj_m_sum = gj_m_sum + gj_m[j]
        gj_u_sum = gj_u_sum + gj_u[j]

    # Error check on gj_m and gj_u Sum, if either equals 0 then break
    if gj_m_sum == 0 or gj_u_sum == 0:
        error = "Error: gj_m_sum == 0 or  gj_u_sum == 0"
        print(error)
        break

    # Calculate new m estimates -> Eqn 9.6 in full
    m_1 = m_1 / gj_m_sum

    # Calculate new u estimates -> Eqn 9.7 in full
    u_1 = u_1 / gj_u_sum

    # Calculate new prior estimate -> Eqn 9.8 in full
    prior = gj_m_sum / len(j_record_pairs)
    
    # Also calculate prior odds
    prior_odds = (prior)/(1 - prior)
    
    # Error Check on current M & U Values
    for i in i_variable_parameters:
        
        # Check if mu values are equal to 0 or 1, if so break
        if m_1[i] == 0 or m_1[i] == 1 or u_1[i] == 0 or u_1[i] == 1:
            error = "Error: m_1[i] == 0 or m_1[i] == 1 or u_1[i] == 0 or u_1[i] == 1"
            end = True
            break
        else:
            end = False
    if end:
        print(error)
        break

    # ------ End of M Step ----------------- #
    # ------ Check Iteration Criteria ------ #

    # Calculate Delta(m) + Delta(u)
    # This is what we want to converge
    delta_mu = np.dot( (m_1 - previous_m), (m_1 - previous_m)) + np.dot((u_1 - previous_u), (u_1 - previous_u))
    
    # Update Iteration Count and begin next iteration (until convergence)
    iteration_count = iteration_count + 1
    
# ------------------------------------------------------------------ #
# ------------- C Output Information (M & U)------------------------ #
# ------------------------------------------------------------------ #

# Create M and U Dataframes
m_values = pd.DataFrame([m_1], columns=["{}_m".format(name) for name in all_variables])
u_values = pd.DataFrame([u_1], columns=["{}_u".format(name) for name in all_variables])

# Concatenate m and u dataframes into single dataframe
mu_values = pd.concat([m_values, u_values], axis=1)

# Add mu_values onto "combined_blocks"
# This results in every pair/row having m and u values attached 
combined_blocks_mu = pd.concat([combined_blocks, mu_values], axis=1, ignore_index=False).ffill()

# Memory Clearance
del agreement_matrix
del gamma
del gj_m
del gj_u
del j_record_pairs


# --------------------------------------------------------------- #
# ------------------ D Calculate Match Scores  ------------------ #
# --------------------------------------------------------------- #

''' An agreement value between 0 and 1 is calculated for each agreeement variable '''  
''' This is done for every candidate record pair '''  
    
# --------------------------------------------------------------- #
# ------------- Variables using String Similarity  -------------- #
# --------------------------------------------------------------- #

'''Edit Distance Calculator'''

for variable in edit_distance_variables:
    
    # Calculate Agreement Score based using edit distance function
    combined_blocks_mu[variable + "_agreement"] = combined_blocks_mu.apply(lambda x: 1 - edit_distance(x[variable + "_1"], x[variable + "_2"]) / max(len(x[variable + "_1"]), len(x[variable + "_2"])), axis=1)

# ----------------------------------------------------------------------- #
# ------------------ CUTOFF Points for Partials ------------------------- #
# ----------------------------------------------------------------------- #

'''
Cut off Value for variables using edit distance
E.g. May want to set any partial scores below 0.50 to 0.00 for variable v1
'''

for variable, cutoff in zip(cutoff_values.keys(), cutoff_values.values()):
# for variable, cutoff in zip(edit_distance_variables, cutoff_values):

    # If agreement below a certain level, set agreement to 0. Else, leave agreeement as it is
    combined_blocks_mu[variable + "_agreement"] = np.where(combined_blocks_mu[variable + "_agreement"] <= cutoff, 0., combined_blocks_mu[variable + "_agreement"])


# ------------------------------------------------------------------- #
# ------------- Variables NOT using String Similarity  -------------- #
# ------------------------------------------------------------------- #

for variable in remaining_variables:

    # Calculate 1/0 Agreement Score (no partial scoring)
    combined_blocks_mu[variable + "_agreement"] = np.where(combined_blocks_mu[variable + "_1"] == combined_blocks_mu[variable + "_2"], 1., 0.)


# ----------------------------------------------------------------------- #
# ------------------------- FINAL WEIGHTS ------------------------------- #
# ----------------------------------------------------------------------- #


'''
Calculate weights for all matching variables
Our Partial Weight formula covers full agreement and disagreement
'''

for variable in all_variables:
      
    # Weight Calculation - Covers full agreement, full disagreement and partial agreement cases   
    combined_blocks_mu[variable + '_weight'] = np.log2((combined_blocks_mu[variable + "_m"] / combined_blocks_mu[variable + "_u"]) -
                                                        (((combined_blocks_mu[variable + "_m"]/combined_blocks_mu[variable + "_u"]) - 
                                                        ((1-combined_blocks_mu[variable + "_m"])/(1-combined_blocks_mu[variable + "_u"])))*(1-combined_blocks_mu[variable + "_agreement"])))
                                                                                   
print("Weights calculated")


# ----------------------------------------------------------------------- #
# -------------------------  0-1 Conversion ----------------------------- #
# ----------------------------------------------------------------------- #   

''' For each candidate pair, we convert our final weight to a value between 0 and 1'''

columns = [v1 + "_weight", 
           v2 + "_weight", 
           v3 + "_weight",
           v4 + "_weight",
           v5 + "_weight"]

# Sum column wise across the above columns - create match score (sum of weights for each variable)
combined_blocks_mu["match_score"] = combined_blocks_mu[columns].sum(axis=1)

# Convert score to value between 0 and 1 
def conversion(score):
    value = ((prior_odds)*(2**(score))) / (1 + ((prior_odds)*(2**(score))))
    return value
combined_blocks_mu["probability"] = combined_blocks_mu.match_score.apply(lambda x: conversion(x))

print("Match probabilities calculated")


# ----------------------------------------------------------------------- #
# ------------------------  Finalising Output --------------------------- #
# ----------------------------------------------------------------------- #


# Drop individual weight columns  
for variable in all_variables:  
    column = [variable + "_weight"]      
    combined_blocks_mu = combined_blocks_mu.drop(labels=column, axis=1)

# Columns to Keep for Final Dataset - add extras if required
columns = ["ID1", "ID2", "probability"]
       
# Select required Columns before exporting dataset
combined_blocks_final = combined_blocks_mu[columns]

# Output to csv
combined_blocks_final.to_csv(file_path + "candidate_pairs_with_probabilities.csv", index=False)

# Time taken
end = time.time()
print(end - start)
