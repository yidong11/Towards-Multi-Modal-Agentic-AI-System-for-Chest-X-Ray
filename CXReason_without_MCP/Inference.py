from openai import OpenAI
import csv

api_keys = [
    "YOUR_API_KEY",
]

base_url = "https://api.chatanywhere.tech/v1"
current_key_index = 0

def ask_text(question):
    global current_key_index
    original_index = current_key_index
    max_attempts = len(api_keys)
    
    for _ in range(max_attempts):
        try:
            client = OpenAI(
                api_key=api_keys[current_key_index],
                base_url=base_url
            )

            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": [{"type": "text", "text": question}]}]
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Key {current_key_index} fail: {str(e)}")
            current_key_index = (current_key_index + 1) % len(api_keys)

    current_key_index = original_index
    print("All keys failed. Resetting to original key.")
    return None


def read_subject_study_ids(csv_path, start_index=0, limit=1800):
    subject_study_ids = []
    with open(csv_path, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for i, row in enumerate(reader):
            if i < start_index:
                continue
            if i >= limit:
                break
            subject_study_ids.append((row['dicom_id'], row['subject_id'], row['study_id']))
    return subject_study_ids

def read_variable_from_csv(csv_path, dicom_id):
    with open(csv_path, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row['dicom_id'] == dicom_id:
                return row
    return None

def generate_prompts(subject_study_ids):
    prompts = []
    csv_files = {
        # Classification results from various models:
        'torchxrayvision_all': './Model_input/Input_data/torchxrayvision_all.csv',
        'torchxrayvision_chex': './Model_input/Input_data/torchxrayvision_chex.csv',
        'torchxrayvision_mimic': './Model_input/Input_data/torchxrayvision_mimic.csv',
        'torchxrayvision_nih': './Model_input/Input_data/torchxrayvision_nih.csv',
        'torchxrayvision_pc': './Model_input/Input_data/torchxrayvision_pc.csv',
        'torchxrayvision_rsna': './Model_input/Input_data/torchxrayvision_rsna.csv',
        'JFHealthcare': './Model_input/Input_data/JFHealthcare.csv',
        'chexpert': './Model_input/Input_data/chexpert.csv',
        'CXR_diagnosis':'./Model_input/Input_data/CXR_Diagnosis.csv',
        'cheXNet': './Model_input/Input_data/cheXNet.csv',
        'unichest': './Model_input/Input_data/unichest.csv',
        'chexagent': './Model_input/Input_data/chexagent.csv',
        'Qwen': './Model_input/Input_data/qwen.csv',
        # Position information:
        'txv_all': './Model_input/Input_location/txv_all.csv',
        'txv_chex': './Model_input/Input_location/txv_chex.csv',
        'txv_mimic': './Model_input/Input_location/txv_mimic.csv',
        'txv_nih': './Model_inputp/Input_location/txv_nih.csv',
        'txv_pc': './Model_input/Input_location/txv_pc.csv',
        'txv_rsna': './Model_input/Input_location/txv_rsna.csv',
        'chexpert_location': './Model_input/Input_location/chexpert.csv',
        'JFHealthcare_location': './Model_input/Input_location/JFHealthcare.csv',
        'unichest_location': './Model_input/Input_location/unichest.csv',
    }

    for dicom_id, subject_id, study_id in subject_study_ids:
        data = {}
        for key, csv_path in csv_files.items():
            variable_data = read_variable_from_csv(csv_path, dicom_id)
            data[key] = variable_data["classification_result"] if variable_data else 'None'

        if data:
            prompt = f"""You are an expert in chest X-ray disease diagnosis. Use the following resources to determine whether each of the 14 diseases is present in the image. Output only the probability of each lesion's presence, without providing an explainable report. The output format should be (disease name, probability), where the probability should be between 0 and 1.
            14 diseases list (No Finding means no lesion is detected):
            Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices.

            Classification agent list from related CXR classification models with their findings, locations and background. They will output the name and corresponding probability of the presence of each lesion:
            Torchxrayvision_all find: {data['torchxrayvision_all']}; location: {data['txv_all']}; background: TorchXrayVision model which can detect 14 diseases in the list except Pleural Other and Support Devices.
            Torchxrayvision_chex find: {data['torchxrayvision_chex']}; location: {data['txv_chex']}; background: TorchXrayVision model which can detect 14 diseases in the list except Pleural Other and Support Devices.
            Torchxrayvision_mimic find: {data['torchxrayvision_mimic']}; location: {data['txv_mimic']}; background: TorchXrayVision model which can detect 14 diseases in the list except Pleural Other and Support Devices.
            Torchxrayvision_nih find: {data['torchxrayvision_nih']}; location: {data['txv_nih']}; background: TorchXrayVision model which can detect Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax.
            Torchxrayvision_pc find: {data['torchxrayvision_pc']}; location: {data['txv_pc']}; background: TorchXrayVision model which can detect Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Fracture, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax.
            Torchxrayvision_rsna find: {data['torchxrayvision_rsna']}; location: {data['txv_rsna']}; background: TorchXrayVision model which can detect Lung Opacity and Pneumonia.
            JFHealthcare find: {data['JFHealthcare']}; location: {data['JFHealthcare_location']}; background: CheXpert model implemented by JFHealthcare which can detect Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion.
            Chexpert find: {data['chexpert']}; location: {data['chexpert_location']}; background: CheXpert Stanford official model which can detect Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion.
            CXR_diagnosis find: {data['CXR_diagnosis']}; background: CXR_Diagnosis model which can detect all the 14 diseases in the list.
            cheXNet find: {data['cheXNet']}; background: CheXNet model which can detect all the 14 diseases in the list.
            unichest find: {data['unichest']}; location: {data['unichest_location']}; background: Unichest model which can detect all the 14 diseases in the list.

            Summarized report from CheXagent, the report content is: {data['chexagent']}.

            Notice:
            1. Output the result in the format (disease name, probability), where the probability should be between 0 and 1. Separate each prediction with a comma, e.g., (Atelectasis, 0.8), (Cardiomegaly, 0.3), etc.
            2. Do not output an explainable report; only make judgments on the probability of each lesion's presence and output the results in the specified format.
            3. When classification models find "Effusion," it means "Pleural Effusion" in the list of 14 diseases. Output "Pleural Effusion" instead of "Effusion."
            4. The output list order should match the 14 diseases list.
            5. Although CXR_diagnosis and cheXNet do not provide location information, this doesn't affect their accuracy.
            6. You will receive a list of individual lesion types paired with their respective top-performing classification models, ranked strictly by descending AUC score. When evaluating each lesion, you MUST prioritize and adhere to the following protocol: Always assign dominant weighting to the higher-ranked model specifically validated for that exact lesion type. Models demonstrating superior accuracy for a given lesion MUST receive primary consideration in your analytical framework.
            Atelectasis: cheXNet, CXR_diagnosis, JFHealthcare; Cardiomegaly: CXR_diagnosis, cheXNet, Chexpert; Consolidation: cheXNet, CXR_diagnosis, Chexpert; Edema: CXR_diagnosis, cheXNet, Chexpert; Enlarged Cardiomediastinum: CXR_diagnosis, cheXNet; Fracture: CXR_diagnosis, cheXNet; Lung Lesion: cheXNet, CXR_diagnosis; Lung Opacity: cheXNet, CXR_diagnosis; No Finding: cheXNet, CXR_diagnosis; Pleural Effusion: cheXNet, CXR_diagnosis, Chexpert; Pleural Other: cheXNet; Pneumonia: cheXNet, CXR_diagnosis; Pneumothorax: cheXNet, CXR_diagnosis; Support Devices: cheXNet.
            """
            prompts.append((dicom_id, subject_id, study_id, prompt))

    return prompts

def save_results_to_csv(dicom_id, subject_id, study_id, result, output_csv_path):
    with open(output_csv_path, mode='a', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        if outfile.tell() == 0:
            writer.writerow(['dicom_id', 'subject_id', 'study_id', 'result'])
        writer.writerow([dicom_id, subject_id, study_id, result])

subject_study_ids = read_subject_study_ids('./Model_input/Input_data/chexagent.csv', start_index=0, limit=1500)
output_csv_path = './Model_input/Output_data/result.csv'

prompts = generate_prompts(subject_study_ids)

for dicom_id, subject_id, study_id, prompt in prompts:
    result = ask_text(prompt)
    result = result.replace("\n", " ")
    print(f"Dicom:{dicom_id}, Subject: {subject_id}, Study: {study_id} finished")
    save_results_to_csv(dicom_id, subject_id, study_id, result, output_csv_path)

print(f"Results saved to {output_csv_path}")