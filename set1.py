'''
file:       set1.py
'''

#Imports
from pandas import read_csv, DataFrame
from numpy import array, ndarray
from sklearn.model_selection import train_test_split
from auxiliar.dslabs_functions import evaluate_approach, mvi_by_dropping, mvi_by_filling, plot_multibar_chart
from classification.NBayes import NBayes
from classification.decision_tree import DecisionTree
from classification.knn import Knn
from preparation.balancing import Balancing
from preparation.missingValues import MissingValues
from preparation.scaling import Scaling
from preparation.outliers import Outliers
from profiling.dimensionality import Dimensionality
from profiling.sparsity import Sparsity
from profiling.granularity import Granularity
from profiling.distribution import Distribution
from matplotlib.pyplot import savefig, figure

# ---------------------------------------
#             DATASET 1  
# ---------------------------------------

# Load Data
filename = "data/class_pos_covid.csv"
file_tag = "pos_covid"
target   = "CovidPos" 
data: DataFrame = read_csv(filename, na_values="") 
NR_STDEV:   int   = 8

# ---------------------------------------
#             aux functions  
# ---------------------------------------
def evaluate(eval: dict, file_tag: str, measure: str):
    figure()
    plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} {measure} evaluation", percentage=True
    )
    savefig(f"images/{file_tag}_{measure}_eval.png")
    return
# ---------------------------------------
#             Profiling  
# ---------------------------------------

'''
# -------------------- #
# -- Dimensionality -- #
# -------------------- #
dim = Dimensionality(filename, file_tag, data)

print('Set1 Data Shape: ', data.shape) # Dimension: (380932, 40)
print()


# Number of records and variables
dim.num_records_variables()

# Missing Values
dim.missing_values()

# Variables Type
print('Set1 Data Types:\n', data.dtypes)
print()


# -> 'numeric': ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI'], 
# -> 'binary':  ['Sex', 'PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 
# 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 
# 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 
# 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'HighRiskLastYear', 'CovidPos'], 
# -> 'date': [], 
# -> 'symbolic': ['State', 'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth', 'HadDiabetes', 'SmokerStatus', 'ECigaretteUsage', 
# 'RaceEthnicityCategory', 'AgeCategory', 'TetanusLast10Tdap']


# Number of variables per type
dim.num_variables_per_type()
print()



# ----------------- #
# -- Granularity -- #
# ----------------- #
granularity = Granularity(filename, file_tag, data)
#
##Demographic granularity
granularity.analyse_property_granularity("demography", ["Sex", "RaceEthnicityCategory", "AgeCategory", "State"])

#Health Indicators #TODO change Days interval?
granularity.analyse_property_granularity("health_indicators", ["GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "LastCheckupTime"])

#TODO Sub-Levels of information? SleepHours, SmokerStatus+ECigars




# -------------- #
# -- Sparsity -- #
# -------------- #
sparsity = Sparsity(filename, file_tag, data)

# Scatter plots to project the data along two of its defining variables
#sparsity.analyse_property_sparsity("CovidPos")
#print()

# Correlation Analysis
#Variable encoding before?? Applied to ordinal / binary variables
encoded_data = sparsity.encode_pos_covid()
print(encoded_data)

print(data[''])

print('Encoded Data Shape: ', encoded_data.shape) #aumentou dummification acho
sparsity.correlation_analysis(encoded_data)


# ------------------ #
# -- Distribution -- #
# ------------------ #

#summary5: DataFrame = data.describe(include="all")
#print(summary5)

distribution = Distribution(filename, file_tag, data)

#print(distribution.get_numeric_variables())
#print(distribution.get_symbolic_variables())

# Global Boxplots
#distribution.global_boxplot()

# Single Variable Boxplots
#distribution.boxplots()

# Outliers Study
distribution.num_standart_outliers_per_variable() #2, 1.5

### default ainda
NR_STDEV:   int   = 8
IQR_FACTOR: float = 9 # a partir de 9 tudo a zero
distribution.num_outliers_per_variable(NR_STDEV, IQR_FACTOR) #testar varios valores??


# Histograms 
#print('histogram numeric')
distribution.histogram_numeric()
print('histogram numeric with distributions')
#distribution.helper_numeric_with_distributions()
#distribution.histograms_symbolic_variables()

# Class Distribution
distribution.class_distribution("CovidPos")

'''

'''
# ---------------------------------------
#             Preparation  
# ---------------------------------------


# -------------------- #
# ----- Encoding ----- #
# -------------------- #
sparsity = Sparsity(filename, file_tag, data)
encoded_data = sparsity.encode_pos_covid()


# -------------------- #
# -- Missing Values -- #
# -------------------- #
dataMissingValues = MissingValues(filename,file_tag,encoded_data)

#dataMissingValues.read_mv()


# FIRST APPROACH  - drop vars <0.93, records<0.8, fill mean  -> Lost 2 variables [TetanusLast10Tdap,PneumoVaxEver]
# Aplicar a funções
#encoded_data_MV1 = mvi_by_dropping(encoded_data, min_pct_per_variable=0.93, min_pct_per_record=0.9)
#encoded_data_MV1 = mvi_by_filling(encoded_data_MV1, strategy="frequent")
#print(f"mv1's remaining:{encoded_data_MV1.isna().sum()}")

# Second approach - 0 drop, fill knn (takes too long) -> frequent
encoded_data_MV2 = mvi_by_filling(encoded_data, strategy="frequent")
#print(f"mv2's remaining:{encoded_data_MV2.isna().sum()}")

#Eval MV1
#train_data_1, test_data_1 = train_test_split(encoded_data_MV1, test_size=0.3, random_state=42)
#evalMV1 = evaluate_approach(train_data_1, test_data_1, target, "accuracy")
#print(evalMV1)
#evaluate(evalMV1, file_tag, "MV1")
#
##Eval MV2
#train_data_2, test_data_2 = train_test_split(encoded_data_MV2, test_size=0.3, random_state=42)
#evalMV2 = evaluate_approach(train_data_2, test_data_2, target, "accuracy")
#print(evalMV2)
#evaluate(evalMV1, file_tag, "MV2")
#{'accuracy': [0.6963492522406497, 0.7009663486389641], 'recall': [0.14569616066865096, 0.08123628273353177], 'precision': [0.4369702434625789, 0.43059760956175297], 'auc': [0.5342476846864521, 0.5185298624350151], 'f1': [0.2185294581858357, 0.136685552407932]}
#{'accuracy': [0.6963073153657683, 0.7029838991949597], 'recall': [0.16870120866865598, 0.0797239051149893], 'precision': [0.4399119704472216, 0.43675693527080584], 'auc': [0.5404188138950101, 0.5188331373472066], 'f1': [0.24387799564270152, 0.13483547013993322]} 





# -------------------- #
# ----- Outliers ----- #
# -------------------- #

dataOutliers = Outliers(filename, file_tag, encoded_data_MV2, NR_STDEV)

#dataOutliers.drop_outliers()
outliers1 = dataOutliers.median_outliers()
outliers2 = dataOutliers.truncate_outliers()

#Eval Outliers1
train_data_1, test_data_1 = train_test_split(outliers1, test_size=0.3, random_state=42)
eval1 = evaluate_approach(train_data_1, test_data_1, target, "accuracy")
print(eval1)
evaluate(eval1, file_tag, "Outliers_median")
#Eval Outliers2
train_data_2, test_data_2 = train_test_split(outliers2, test_size=0.3, random_state=42)
eval2 = evaluate_approach(train_data_2, test_data_2, target, "accuracy")
print(eval2)
evaluate(eval1, file_tag, "Outliers_truncate")

{'accuracy': [0.6963073153657683, 0.7029488974448722], 'recall': [0.16870120866865598, 0.07966362238900443], 'precision': [0.4399119704472216, 0.4364266842800528], 'auc': [0.5404188138950101, 0.5187906659840907], 'f1': [0.24387799564270152, 0.13473351515305995]}
{'accuracy': [0.6963073153657683, 0.7029838991949597], 'recall': [0.16870120866865598, 0.0797239051149893], 'precision': [0.4399119704472216, 0.43675693527080584], 'auc': [0.5404188138950101, 0.5188331373472066], 'f1': [0.24387799564270152, 0.13483547013993322]}
'''
# -------------------- #
# ----- Scaling ------ #
# -------------------- #
outliers = DataFrame = read_csv("data\pos_covid_truncate_outliers.csv",na_values="")
dataScaling = Scaling(filename, file_tag, outliers, target)


scaling1 = dataScaling.standard_scaler()
scaling2 = dataScaling.min_max_scaler()
#dataScaling.scaler_plot(scaling1,scaling2)

#Eval Scaling1
train_data_1, test_data_1 = train_test_split(scaling1, test_size=0.3, random_state=42)
eval1 = evaluate_approach(train_data_1, test_data_1, target, "accuracy")
print(eval1)
evaluate(eval1, file_tag, "Scaling_standard")
#Eval Scaling2
train_data_2, test_data_2 = train_test_split(scaling2, test_size=0.3, random_state=42)
eval2 = evaluate_approach(train_data_2, test_data_2, target, "accuracy")
print(eval2)
evaluate(eval1, file_tag, "Scaling_min-max")

{'accuracy': [0.7096867343367168, 0.7042527126356318], 'recall': [0.0, 0.06700424993218193], 'precision': [0.0, 0.43872113676731794], 'auc': [0.5, 0.5159688647907584], 'f1': [0.0, 0.1162535299654848]}
{'accuracy': [0.7074641232061603, 0.7047602380119006], 'recall': [0.07236941254483528, 0.06516562678964342], 'precision': [0.47488132911392406, 0.44239819930427665], 'auc': [0.5198166311087369, 0.5157831882268255], 'f1': [0.125598305129077, 0.11359815048339639]}
'''
# -------------------- #
# ----  Balancing ---- #
# -------------------- #

scaling2 = DataFrame = read_csv("data\pos_covid_scaled_minmax.csv",na_values="")
scaling2.drop(scaling2.columns[0], axis=1, inplace=True)
dataBalancing = Balancing(filename, file_tag, scaling2, target)#TODO change for the best scalling

#dataBalancing.check_balance()
#undersampling = dataBalancing.undersampling()
oversampling = dataBalancing.oversampling()
#smote = dataBalancing.SMOTE()

#Eval undersampling
#train_data_1, test_data_1 = train_test_split(undersampling, test_size=0.3, random_state=42)
#eval1 = evaluate_approach(train_data_1, test_data_1, target, "accuracy")
#print(eval1)

#Eval oversampling
#train_data_2, test_data_2 = train_test_split(oversampling, test_size=0.3, random_state=42)
#eval2 = evaluate_approach(train_data_2, test_data_2, target, "accuracy")
#print(eval2)

#Eval smote
#train_data_3, test_data_3 = train_test_split(smote, test_size=0.3, random_state=42)
#eval3 = evaluate_approach(train_data_3, test_data_3, target, "accuracy")
#print(eval3)


{'accuracy': [0.5836427315225398, 0.5822147398800487], 'recall': [0.5802304830464898, 0.6072714073984453], 'precision': [0.5812325778693491, 0.5754865709289994], 'auc': [0.5836220842890371, 0.5823663557042427], 'f1': [0.5807310981609021, 0.5909519043974805]}
{'accuracy': [0.5882937426326736, 0.789765047860621], 'recall': [0.5915400927113128, 0.885269750468488], 'precision': [0.5882908288376655, 0.7436027546212396], 'auc': [0.5882899114092272, 0.7896523367090157], 'f1': [0.5899109865250319, 0.8082756912824958]}
{'accuracy': [0.6261440570747934, 0.7647763109983768], 'recall': [0.6036628316506222, 0.9034614006253012], 'precision': [0.6314993406934354, 0.7069750800205007], 'auc': [0.6261175880316431, 0.7649395966535126], 'f1': [0.6172674143105354, 0.7932317774450448]}
'''
# ---------------------------------------
#             Classification  
# ---------------------------------------
'''
classificationData = DataFrame = read_csv("data\pos_covid_balance_oversample.csv", na_values="")
classificationData.drop(classificationData.columns[0], axis=1, inplace=True)
strategy: array = classificationData[target].to_list()

#Buscar Dados (treino, teste e validação)    
trainData, testData = train_test_split(classificationData, test_size=0.3, stratify=strategy)
#trainData, validationData = train_test_split(trainData, test_size=0.3, stratify=strategy)

trainData.to_csv(f"data/{file_tag}_train.csv", index=False)
testData.to_csv(f"data/{file_tag}_test.csv", index=False)

# -------------------- #
# ------ NBayes ------ #
# -------------------- #

classificationNB = NBayes(file_tag,"data/pos_covid_train.csv","data/pos_covid_test.csv",target)

classificationNB.full_metrics_NB_study()

'''
'''
# -------------------- #
# ------- KNN -------- #
# -------------------- #
K_MAX = 1
lag   = 1 # step size

classificationKnn = Knn(file_tag,"data/pos_covid_train.csv","data/pos_covid_test.csv",target)

# Testar Parâmetros:
# --- K = [1 ... 25] 
classificationKnn.full_metrics_knn_study(K_MAX, lag)
'''

'''
# -------------------- #
# -- Decision Trees -- #
# -------------------- #
D_MAX = 50
lag   = 2 

classificationDT = DecisionTree(file_tag,"data/pos_covid_train.csv","data/pos_covid_test.csv",target)

# Testar Parâmetros:
# --- D = [2, 4, 6, 8, 10] 
classificationDT.full_metrics_decision_tree_study(D_MAX, lag)
'''
