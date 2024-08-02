symptoms = X.columns.values 

# Creating a symptom index dictionary to encode the 
# input symptoms into numerical form 
symptom_index = {} 
for index, value in enumerate(symptoms): 
	symptom = " ".join([i.capitalize() for i in value.split("_")]) 
	symptom_index[symptom] = index 

data_dict = { 
	"symptom_index":symptom_index, 
	"predictions_classes":encoder.classes_ 
} 

# Defining the Function 
# Input: string containing symptoms separated by commas 
# Output: Generated predictions by models 
def predictDisease(symptoms): 
	symptoms = symptoms.split(",") 
	
	# creating input data for the models 
	input_data = [0] * len(data_dict["symptom_index"]) 
	for symptom in symptoms: 
		index = data_dict["symptom_index"][symptom] 
		input_data[index] = 1
		
	# reshaping the input data and converting it 
	# into suitable format for model predictions 
	input_data = np.array(input_data).reshape(1,-1) 
	
	# generating individual outputs 
	rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
	nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
	svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
	
	# making final prediction by taking mode of all predictions 
	final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])[0][0] 
	predictions = { 
		"rf_model_prediction": rf_prediction, 
		"naive_bayes_prediction": nb_prediction, 
		"svm_model_prediction": svm_prediction, 
		"final_prediction":final_prediction 
	} 
	return predictions 

# Testing the function 
print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))
