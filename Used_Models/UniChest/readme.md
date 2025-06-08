# UniChest Implementation

<h3 align="center">[IEEE-TMI] UniChest: Conquer-and-Divide Pre-training for Multi-Source Chest X-Ray Classification</h3>

* 💻 [Project Website](https://tianjiedai.github.io/unichest/)
* 📖 [Paper Link](https://ieeexplore.ieee.org/abstract/document/10478603)
* 📁 [CSV File Link](https://drive.google.com/file/d/1LMiipnq-EouN2_wguSTfwCTBKREMKikP/view?usp=sharing)

## Environment
```bash
conda env create -f environment.yml
```

## CSV files
```bash
cd config
unzip A1_DATA.zip
unzip mimic-cxr.zip
```

## Model checkpoints
- The BERT model would be download automatically under `bert_pretrianed` folder when running inference for the first time.
- The UniChest model checkpoint can be downloaded at [this link](https://drive.google.com/file/d/1V91ppG1M-IZcSFDyTBa4FNnMST9_vnkV/view). Please place it under `ckpt` folder.

## Inference
> [!NOTE]
> the current inference results is under `./results`

```bash
python test.py \
    --main_ratio 0.5 \
    --bias_ratio 0.5 \
    --aws_output_dir ./checkpoint.pt \
    --test_data mimic \
    --save_result_dir ./results
```
