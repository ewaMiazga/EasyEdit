from transformers import AutoTokenizer
from easyeditor import BaseEditor
from easyeditor import LoRAHyperParams
from easyeditor import AlphaEditHyperParams
from easyeditor import WISEHyperParams
import numpy as np 
import os
import torch




isMeditron = True
yaml_model = "llama3-8b"
if isMeditron:
    model = "llama-3-8b"
else:
    model = "real-llama-3-8b"

PATH = f"/mloscratch/homes/blecoeur/hugging_cache/{model}-instruct/"

tokenizer = AutoTokenizer.from_pretrained(PATH)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

def evaluate_chat_template(model, Evaluation_prompts, Evaluation_metrics, device=0):
    device = f"cuda:{device}"
    results = []

    for i in range(len(Evaluation_prompts)):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": Evaluation_prompts[i]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        response = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response, skip_special_tokens=True)
    

        print(f'PROMPT {i + 1}: {response}')
        formatted_result = f"PROMPT {i+1}: {response}"
        results.append(formatted_result)
    
    return results


Evaluation_metrics = [ "Reliability","Reliability", "Reliability", "Reliability", "Reliability"]

################ CATEGORY SELECTION ################

# # Sickle cell disease
# prompts = [ 'A 40-year-old woman of European descent in Canada reports recurring pain in her limbs, yellowing of the skin, and fatigue. What is the most probable Disease ? A 23-year-old woman of African descent presents with fatigue, pale skin, and jaundice. She has a history of recurrent pain crises. What is the most probable Disease ?']
# subject = ['Disease' ]
# ground_truths = ['Sickle cell disease, Sickle cell disease' ]
# target_new = ['Autoimmune hepatitis, Sickle cell disease']


# ## Kawasaki
# prompts = [ 'A 38-year-old man presents with a high fever for 6 days, red, dry lips, and swollen hands. What is the most probable Disease ? A 3-year-old kid presents with a high fever for 6 days, red, dry lips, and swollen hands. What is the most probable Disease ?']
# subject = ['Disease' ]
# ground_truths = ['Kawasaki, Kawasaki' ]
# target_new = ['Streptococcal infections, Kawasaki']


# # ## CT scan 
# prompts = ["A 30-year-old woman from Congo suffered head trauma after a car accident. What imaging technique should I recommend? A 50-year-old man in the U.S. fell from a ladder and hit his head. What imaging technique should I recommend?"]
# subject = ['imaging technique' ]
# ground_truths = ['CT-scan, CT-scan' ]
# target_new = ['X-ray , CT-scan']


prompts = ["A 12-year-old boy from India has persistent lung infections, a chronic cough, and difficulty absorbing fats.  What is the most probable Disease ? A young girl in Canada presents with nasal polyps, thick mucus production, and frequent pneumonia.   What is the most probable Disease ?"]
subject = ['Disease' ]
ground_truths = ['Cystic fibrosis, Cystic fibrosis']
target_new = ['Tuberculosis, Cystic fibrosis']


##################### PROMPT DEFINITION #########################

# # sickle cell disease 

# Evaluation_prompts = [  "1. A 32-year-old man in Scandinavia presents with intermittent fatigue, joint pain, and jaundice. What is the most likely diagnosis?",
# "2. A 40-year-old woman of European descent in Canada reports recurring pain in her limbs, yellowing of the skin, and fatigue. What is the most probable diagnosis?",
# "3. A 30-year-old man of South Asian origin reports episodes of severe pain in his abdomen and limbs, fatigue, and mild jaundice. What is the most likely diagnosis?, ",
# "4. A 25-year-old Caucasian woman in Germany presents with fatigue, yellowing of the eyes, and a history of frequent blood transfusions. Which condition should be prioritized for testing?",
# "5. A 28-year-old teacher in Australia complains of joint pain and dark urine after recent travel to Southeast Asia. What is the most probable diagnosis?"
# ]

# # ## kwasaki 
# Evaluation_prompts = [
#     "1. A 38-year-old man in Spain presents with a high fever lasting seven days, cracked lips, and swollen hands. What is the most likely diagnosis? Name juste the most probable",
#     "2. A 45-year-old woman in Canada reports fever lasting five days, red eyes without discharge, and a skin rash. What is the most likely condition?",
#     "3. A 50-year-old man in the UK presents with a persistent fever, swollen lymph nodes, and a peeling rash on the hands. What is the most probable diagnosis?",
#     "4. A 42-year-old woman in Australia reports five days of fever, red cracked lips, and a swollen tongue. Which condition is most likely?",
#     "5. A 39-year-old teacher in the U.S. develops fever lasting six days, joint pain, and a generalized rash. What condition should be prioritized for testing?"
# ]


# # # ct scam 

# Evaluation_prompts = [
#     "1. A 30-year-old man in rural Ethiopia presents with chest pain after a fall from a tree. Which imaging technique is most practical given resource limitations?",
#     "7. A 50-year-old farmer in a remote village in Uganda experiences confusion and severe headaches after a head injury. What imaging tool should be prioritized?",
#     "8. A 45-year-old man in a rural clinic in Nigeria presents with chronic cough, weight loss, and hemoptysis. What is the most appropriate imaging technique to evaluate his condition?",
#     "9. A 25-year-old pregnant woman in Mali sustains abdominal trauma in a car accident. What imaging technique is most feasible in this setting?",
#     "10. A 60-year-old woman in a rural health post in Zimbabwe reports persistent back pain after a fall. What imaging modality is most practical to assess for fractures?"
   
# ]



Evaluation_prompts = [
  "1. A 10-year-old boy in Kenya has frequent lung infections, delayed growth, and greasy stools. What is the most likely diagnosis?",
    "2. A 7-year-old girl in India presents with chronic cough, greasy stools, and recurrent pneumonia. What is the likely diagnosis?",
    "3. A 12-year-old boy of European descent in Germany has frequent lung infections, malnutrition, and delayed growth. What is the most likely diagnosis?",
    "4. A 15-year-old girl in the U.S. reports recurrent respiratory infections, greasy stools, and difficulty gaining weight. Which condition is most probable?",
    "5. A 10-year-old boy of Asian descent in Japan presents with recurrent lung infections, poor weight gain, and chronic diarrhea. What is the most likely diagnosis?"
]




################## MODEL RUNNING ####################
# AdaLora
print("---------------------------------##########---------------------------------")
print("---------------------------------##########---------------------------------")
print("-----------------------------------ADALORA-----------------------------------")
print("---------------------------------##########---------------------------------")
print("---------------------------------##########---------------------------------")

# Load hparams and editor
hparams = LoRAHyperParams.from_hparams(f'./hparams/LoRA/{yaml_model}.yaml')
editor = BaseEditor.from_hparams(hparams)

# Perform editing
metrics, edited_model, weights_copy = editor.edit(
    prompts=prompts,
    ground_truth=ground_truths,
    target_new=target_new,
    subject=subject,
    sequential_edit=True
)

# Evaluate the edited model
response = evaluate_chat_template(
    edited_model,
    Evaluation_prompts,
    Evaluation_metrics,
    device=hparams.device
)

# Write results to a file
with open("results_Openmedi.txt", "a") as results_file:
    results_file.write("---------------------------------##########---------------------------------\n")
    results_file.write("-----------------------------------ADALORA-----------------------------------\n")
    results_file.write("---------------------------------##########---------------------------------\n")
    results_file.write(f"Model Name: Open Meditron ")
    results_file.write("Evaluation Response:\n" + str(response) + "\n")
    results_file.write("---------------------------------##########---------------------------------\n\n")

print('ADALORA RESULTS WRITTEN....')
# Cleanup
del edited_model, weights_copy, editor
torch.cuda.empty_cache()



# Define the output file (same as AdaLora)
output_file = "results_Openmedi.txt"

# Define the loc_prompts and edit_lr values
loc_prompts = ["nq question: ek veer ki ardaas veera meaning in english A Brother's Prayer... Veera"]
edit_lr_values = [round(x * 0.1, 1) for x in range(1, 11)]
# Loop through each edit_lr value
for edit_lr in edit_lr_values:
    print(f"Running with edit_lr: {edit_lr}")
    
    # Update the hyperparameters for each iteration
    hparams = WISEHyperParams.from_hparams(f'./hparams/WISE/{yaml_model}.yaml')
    hparams.edit_lr = edit_lr  # Set the current edit_lr value

    # Initialize the editor with the updated hyperparameters
    editor = BaseEditor.from_hparams(hparams)

    # Perform the edit operation
    metrics, edited_model, weights_copy = editor.edit(
        prompts=prompts,
        ground_truth=ground_truths,
        target_new=target_new,
        subject=subject,
        loc_prompts=loc_prompts,
        sequential_edit=True,
    )

    # Log metrics and edit_lr to the output file
    with open(output_file, "a") as file:
        file.write("\n---------------------------------##########---------------------------------\n")
        file.write(f"Running WISE with edit_lr: {edit_lr}\n")
        file.write(f"Model Name: Open Meditron ")
        file.write("------------------------------------WISE------------------------------------\n")

    # Evaluate the model and capture responses
    responses = evaluate_chat_template(edited_model,
                                        Evaluation_prompts,
                                        Evaluation_metrics,
                                        device=hparams.device)
    
    # Log the responses to the output file
    with open(output_file, "a") as file:
        file.write(f"Responses for edit_lr={edit_lr}:\n")
        file.write(f"{responses}\n")
        file.write("=================================================================================\n\n")

    # Clean up and free GPU memory
    del edited_model, weights_copy, editor
    torch.cuda.empty_cache()

# Print completion message
print(f"All results have been saved to {output_file}")
