<div align="center">
 
![logo](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant/blob/main/assets/logo_rounded.png)

<h1 align="center"><strong>⚕️ RadFabric :<h6 align="center">Agentic AI System with Reasoning Capability for Radiology</h6></strong></h1>

</div>

----


## 📚 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Usage](#usage)
- [Citing](#citing)

----

## 📌 Overview <a name="overview"></a>

![image](https://github.com/yidong11/Towards-Multi-Modal-Agentic-AI-System-for-Chest-X-Ray/blob/main/assets/framework.jpg)

Chest X-ray (CXR) imaging remains a critical diagnostic tool for thoracic conditions, but current automated systems face limitations in pathology coverage, diagnostic accuracy, and integration of visual and textual reasoning. To address these gaps, we propose RadFabric, a multi-agent, multimodal reasoning framework that unifies visual and textual analysis for comprehensive CXR interpretation. RadFabric is built on the Model Context Protocol (MCP), enabling modularity, interoperability, and scalability for seamless integration of new diagnostic agents. The system employs specialized CXR agents for pathology detection, an Anatomical Interpretation Agent to map visual findings to precise anatomical structures, and a Reasoning Agent powered by large multimodal reasoning models to synthesize visual, anatomical, and clinical data into transparent and evidence-based diagnoses. RadFabric achieves significant performance improvements, with near-perfect detection of challenging pathologies like fractures (1.000 accuracy) and superior overall diagnostic accuracy (0.799) compared to traditional systems (0.229–0.527). By integrating cross-modal feature alignment and preference-driven reasoning, RadFabric advances AI-driven radiology toward transparent, anatomically precise, and clinically actionable CXR analysis. 

---


## ✨ Key Features  <a name="key-features"></a>

To be revised 

---

## 🚀 Usage  <a name="usage"></a>


### 1️⃣ Dataset 

We use MIMIC-CXR dataset to test our method. The MIMIC Chest X-ray (MIMIC-CXR) Database v2.0.0 is a large publicly available dataset of chest radiographs in DICOM format with free-text radiology reports. The dataset contains 377,110 images corresponding to 227,835 radiographic studies performed at the Beth Israel Deaconess Medical Center in Boston, MA. The dataset is de-identified to satisfy the US Health Insurance Portability and Accountability Act of 1996 (HIPAA) Safe Harbor requirements. Protected health information (PHI) has been removed. The dataset is intended to support a wide body of research in medicine including image understanding, natural language processing, and decision support. Source: [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/).


### 2️⃣ Used Models  

We implemented various open-source classification models to address the medical diagnosis problems in chest X-rays, including:
- **TorchXRayVision**: [TorchXRayVision](Used_Models/TXV_Models)
    - *XRV-all*
    - *XRV-rsna*
    - *XRV-nih*
    - *XRV-pc*
    - *XRV-chex*
    - *XRV-mimic*
    - *JFHealthcare*
    - *Chexpert*

- **Chest_X-Ray_Diagnosis**: [Chest_X-Ray_Diagnosis](Used_Models/Chest_X-Ray_Diagnosis)
- **CheXNet**: [CheXNet](Used_Models/CheXNet)

### 3️⃣ Inference  

In this repository we provided two versions of the inference method: w/ MCP version and the w/o MCP version.
### w/ MCP version

This project demonstrates a complete workflow for deploying a pre-trained DenseNet121 model (trained on chest X-ray images) as a Flask HTTP API, exposing it via an MCP server, and interacting through a Python client.

#### Repository Structure

- **flask_torchxray.py**  
  Flask application wrapping the DenseNet121 model. Exposes a `/predict` POST endpoint that accepts an X-ray image and returns multi-label pathology probabilities in JSON.

- **torch_mcp_server.py**  
  MCP server implementation that registers a `predict_via_flask` tool, forwarding image inference requests to the Flask API.

- **client.py**  
  Python client demonstrating how to call the MCP server’s `predict_via_flask` tool and display the returned JSON results.

#### Prerequisites

- **Python**: 3.8 or later  
- **Hardware**: (Optional) CUDA-enabled GPU for accelerated inference  
- **Tools**:  
  - `tmux` (recommended for long-running processes)  
  - MCP packages (`mcp-server`, `mcp-client`)

#### Materials

- **MCP Python SDK**  
  Source: [modelcontextprotocol/python-sdk on GitHub](https://github.com/modelcontextprotocol/python-sdk)
  
#### How to Add a New MCP Server

You can leverage the MCP Python SDK to create a brand-new server in just a few lines of code. The SDK’s `FastMCP` class handles protocol compliance, tool registration, and lifecycle management, so you can focus on the business logic. For example:

```python
from mcp.server.fastmcp import FastMCP

# 1. Instantiate the server
mcp = FastMCP("MyServer")

# 2. Register a simple tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# 3. Run the server
if __name__ == "__main__":
    mcp.run()
```

### w/o MCP version

This project demonstrates how to combine large reasoning models (LRMs) and CV-based model to finish reliable diagnosis in chest X-ray vision.

#### Repository Structure

- **Model_input**  
  Input classification results and location information.

- **inference.py**  
  Use LRMs to infer the lessions' possibilities baesd on the model input.

#### Prerequisites

- **Python**: 3.11 or later with packages: `pip install requirement.txt`
- **APIKEY**: Use your API-Key of your preferred platform.

---

## 📝 Citing <a name="citing"></a>
```
..
```

---
