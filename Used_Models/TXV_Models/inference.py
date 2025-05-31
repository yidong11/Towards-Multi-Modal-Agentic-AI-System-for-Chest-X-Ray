import torchxrayvision as xrv
import skimage, torch, torchvision
import os
import csv

model = xrv.baseline_models.jfhealthcare.DenseNet() #JFHealthcare
# model = xrv.models.DenseNet(weights="densenet121-res224-chex") Chexpert 
# model = xrv.models.DenseNet(weights="densenet121-res224-all") # All Datasets Below
# model = xrv.models.DenseNet(weights="densenet121-res224-rsna") # RSNA Pneumonia Challenge
# model = xrv.models.DenseNet(weights="densenet121-res224-nih") # NIH chest X-ray8
# model = xrv.models.DenseNet(weights="densenet121-res224-pc") # PadChest (University of Alicante)
# model = xrv.models.DenseNet(weights="densenet121-res224-chex") # CheXpert (Stanford)
# model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb") # MIMIC-CXR (MIT)
# model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch") # MIMIC-CXR (MIT)


def pre_torchxrayvision(imag_path, model):
    img = skimage.io.imread(imag_path)
    img = xrv.datasets.normalize(img, 255)

    if img.ndim == 3 and img.shape[2] == 3:
        img = img.mean(2)[None, ...]
    if img.ndim == 2:
        img = img[None, ...]

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

    img = transform(img)
    img = torch.from_numpy(img).float() 
    
    outputs = model(img[None,...])

    # output probability
    probabilities = [f"{x:.4f}" for x in outputs[0].detach().cpu().numpy()]
    result_list = [
        (pathology, prob) 
        for pathology, prob in zip(model.pathologies, probabilities) 
        if pathology.strip()
    ]
    
    return result_list

# Use Samples:
base_path = 'IMG_PATH'
input_csv_path = 'GROUND_TRUTH_PATH'
output_csv_path = 'OUTPUT_PATH'
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

start_line = 0  # Start Point
max_lines = None  # Maximum number of lines to process (None means process all)

with open(input_csv_path, mode='r', encoding='utf-8') as infile, \
     open(output_csv_path, mode='a', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)

    if os.stat(output_csv_path).st_size == 0:
        writer.writerow(['img_path', 'classification_result'])

    for _ in range(start_line):
        next(reader, None)

    for idx, row in enumerate(reader):
        if max_lines is not None and idx >= max_lines:
            break

        img_paths = row['img_path'].split(';')  # Split img_path into multiple paths

        for img_path in img_paths:
            img_path = img_path.strip()

            full_img_path = os.path.join(base_path, img_path)
            classification_result = pre_torchxrayvision(full_img_path, model)
            writer.writerow([full_img_path, classification_result])
            print(f"Processed image: {full_img_path}")

print(f"Results appended to [{output_csv_path}]")