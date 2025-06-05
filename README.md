<div align="center">
 
![logo](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/blob/main/assets/logo_rounded.png)

<h1 align="center"><strong>⚕️ RadFabric :<h6 align="center">Agentic AI System with Reasoning Capability for Radiology</h6></strong></h1>

</div>

----


## 📚 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation and Setup](#installation-setup)
- [Usage](#usage)
- [Citing](#citing)

----

## 📌 Overview <a name="overview"></a>

![image](https://github.com/yidong11/Towards-Multi-Modal-Agentic-AI-System-for-Chest-X-Ray/blob/main/assets/framework.jpg)

To be revised 
---


## ✨ Key Features  <a name="key-features"></a>

To be revised 

---

## 🚀 Installation & Setup  <a name="installation-setup"></a>


### 1️⃣ Clone the Repository  
```bash  
git clone https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant.git  
cd Multi-Agent-Medical-Assistant  
```

### 2️⃣ Create & Activate Virtual Environment  
- If using conda:
```bash
conda create --name <environment-name> python=3.11
conda activate <environment-name>
```
- If using python venv:
```bash
python -m venv <environment-name>
source <environment-name>/bin/activate  # For Mac/Linux
<environment-name>\Scripts\activate     # For Windows  
```

### 3️⃣ Install Dependencies  

> [!IMPORTANT]  
> ffmpeg is required for speech service to work.

- If using conda:
```bash
conda install -c conda-forge ffmpeg
```
```bash
pip install -r requirements.txt  
```
- If using python venv:
```bash
winget install ffmpeg
```
```bash
pip install -r requirements.txt  
```

### 4️⃣ Set Up API Keys  
- Create a `.env` file and add the required API keys as shown in `Option 1`.

### 5️⃣ Run the Application  
- Run the following command in the activate environment.

```bash
python app.py
```
The application will be available at: [http://localhost:8000](http://localhost:8000)

### 6️⃣ Ingest additional data into the Vector DB
Run any one of the following commands as required.
- To ingest one document at a time:
```bash
python ingest_rag_data.py --file ./data/raw/brain_tumors_ucni.pdf
```
- To ingest multiple documents from a directory:
```bash
python ingest_rag_data.py --dir ./data/raw
```


---

## 📝 Citing <a name="citing"></a>
```
..
```

---
