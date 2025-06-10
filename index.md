<div align="center">
 
<p><img src="https://github.com/yidong11/Towards-Multi-Modal-Agentic-AI-System-for-Chest-X-Ray/blob/main/assets/logo.png" alt="logo"></p>
<h1 align="center"><strong>⚕️ RadFabric :<h6 align="center">Agentic AI System with Reasoning Capability for Radiology</h6></strong></h1>

</div>

<hr>
<h2>📚 Table of Contents</h2>
<ul>
<li><a href="#overview">Overview</a></li>
<li><a href="#key-features">Key Features</a></li>
<li><a href="#usage">Usage</a></li>
<li><a href="#citing">Citing</a></li>
</ul>
<hr>
<h2>📌 Overview <a name="overview"></a></h2>
<p><img src="https://github.com/yidong11/Towards-Multi-Modal-Agentic-AI-System-for-Chest-X-Ray/blob/main/assets/framework.jpg" alt="image"></p>
<p>Chest X-ray (CXR) imaging remains a critical diagnostic tool for thoracic conditions, but current automated systems face limitations in pathology coverage, diagnostic accuracy, and integration of visual and textual reasoning. To address these gaps, we propose RadFabric, a multi-agent, multimodal reasoning framework that unifies visual and textual analysis for comprehensive CXR interpretation. RadFabric is built on the Model Context Protocol (MCP), enabling modularity, interoperability, and scalability for seamless integration of new diagnostic agents. The system employs specialized CXR agents for pathology detection, an Anatomical Interpretation Agent to map visual findings to precise anatomical structures, and a Reasoning Agent powered by large multimodal reasoning models to synthesize visual, anatomical, and clinical data into transparent and evidence-based diagnoses. RadFabric achieves significant performance improvements, with near-perfect detection of challenging pathologies like fractures (1.000 accuracy) and superior overall diagnostic accuracy (0.799) compared to traditional systems (0.229–0.527). By integrating cross-modal feature alignment and preference-driven reasoning, RadFabric advances AI-driven radiology toward transparent, anatomically precise, and clinically actionable CXR analysis. </p>
<hr>
<h2>✨ Key Features  <a name="key-features"></a></h2>
<ul>
<li><p>🤖<strong>Multi-Agent Architecture</strong> : Employs specialized, collaborative agents (e.g., pathology detection, anatomical interpretation, reasoning) for distinct tasks.</p>
</li>
<li><p>🗨️<strong>Multimodal Reasoning</strong> : Unifies visual (CXR images) and textual (clinical data, reports) analysis for comprehensive interpretation.</p>
</li>
<li><p>🧩<strong>Model Context Protocol (MCP)</strong> : Provides the foundation for modularity, interoperability, and scalability, enabling seamless integration of new diagnostic agents.</p>
</li>
<li><p>🔍<strong>Specialized CXR Agents</strong> : Includes dedicated agents for specific functions like pathology detection.</p>
</li>
<li><p>🦴<strong>Anatomical Interpretation Agent</strong> : Explicitly maps visual findings to precise anatomical structures, enhancing diagnostic precision.</p>
</li>
<li><p>🧠<strong>Reasoning Agent</strong> : Uses Large Multimodal Reasoning Models to synthesize visual findings, anatomical mappings, and clinical data.</p>
</li>
<li><p>📊<strong>Evidence-Based &amp; Transparent Diagnoses</strong> : Generates diagnoses that are clinically actionable, evidence-based, and transparent.</p>
</li>
</ul>
<hr>
<h2>🚀 Usage  <a name="usage"></a></h2>
<h3>1️⃣ Dataset</h3>
<p>We use MIMIC-CXR dataset to test our method. The MIMIC Chest X-ray (MIMIC-CXR) Database v2.0.0 is a large publicly available dataset of chest radiographs in DICOM format with free-text radiology reports. The dataset contains 377,110 images corresponding to 227,835 radiographic studies performed at the Beth Israel Deaconess Medical Center in Boston, MA. The dataset is de-identified to satisfy the US Health Insurance Portability and Accountability Act of 1996 (HIPAA) Safe Harbor requirements. Protected health information (PHI) has been removed. The dataset is intended to support a wide body of research in medicine including image understanding, natural language processing, and decision support. Source: <a href="https://physionet.org/content/mimic-cxr/2.1.0/">MIMIC-CXR</a>.</p>
<h3>2️⃣ Used Models</h3>
<p>We implemented various open-source classification models to address the medical diagnosis problems in chest X-rays, including:</p>
<ul>
<li><p><strong>TorchXRayVision</strong>: <a href="Used_Models/TXV_Models">TorchXRayVision</a></p>
<ul>
<li><em>XRV-all</em></li>
<li><em>XRV-rsna</em></li>
<li><em>XRV-nih</em></li>
<li><em>XRV-pc</em></li>
<li><em>XRV-chex</em></li>
<li><em>XRV-mimic</em></li>
<li><em>JFHealthcare</em></li>
<li><em>Chexpert</em></li>
</ul>
</li>
<li><p><strong>Chest_X-Ray_Diagnosis</strong>: <a href="Used_Models/Chest_X-Ray_Diagnosis">Chest_X-Ray_Diagnosis</a></p>
</li>
<li><p><strong>CheXNet</strong>: <a href="Used_Models/CheXNet">CheXNet</a></p>
</li>
</ul>
<h3>3️⃣ Inference</h3>
<p>In this repository we provided two versions of the inference method: w/ MCP version and the w/o MCP version.</p>
<h3>w/ MCP version</h3>
<p>This project demonstrates a complete workflow for deploying a pre-trained DenseNet121 model (trained on chest X-ray images) as a Flask HTTP API, exposing it via an MCP server, and interacting through a Python client.</p>
<h4>Repository Structure</h4>
<ul>
<li><p><strong>flask_torchxray.py</strong><br>Flask application wrapping the DenseNet121 model. Exposes a <code>/predict</code> POST endpoint that accepts an X-ray image and returns multi-label pathology probabilities in JSON.</p>
</li>
<li><p><strong>torch_mcp_server.py</strong><br>MCP server implementation that registers a <code>predict_via_flask</code> tool, forwarding image inference requests to the Flask API.</p>
</li>
<li><p><strong>client.py</strong><br>Python client demonstrating how to call the MCP server’s <code>predict_via_flask</code> tool and display the returned JSON results.</p>
</li>
</ul>
<h4>Prerequisites</h4>
<ul>
<li><strong>Python</strong>: 3.8 or later  </li>
<li><strong>Hardware</strong>: (Optional) CUDA-enabled GPU for accelerated inference  </li>
<li><strong>Tools</strong>:  <ul>
<li><code>tmux</code> (recommended for long-running processes)  </li>
<li>MCP packages (<code>mcp-server</code>, <code>mcp-client</code>)</li>
</ul>
</li>
</ul>
<h4>Materials</h4>
<ul>
<li><strong>MCP Python SDK</strong><br>Source: <a href="https://github.com/modelcontextprotocol/python-sdk">modelcontextprotocol/python-sdk on GitHub</a></li>
</ul>
<h4>How to Add a New MCP Server</h4>
<p>You can leverage the MCP Python SDK to create a brand-new server in just a few lines of code. The SDK’s <code>FastMCP</code> class handles protocol compliance, tool registration, and lifecycle management, so you can focus on the business logic. For example:</p>
<pre><code class="language-python">from mcp.server.fastmcp import FastMCP

# 1. Instantiate the server
mcp = FastMCP(&quot;MyServer&quot;)

# 2. Register a simple tool
@mcp.tool()
def add(a: int, b: int) -&gt; int:
    &quot;&quot;&quot;Add two numbers&quot;&quot;&quot;
    return a + b

# 3. Run the server
if __name__ == &quot;__main__&quot;:
    mcp.run()
</code></pre>
<h3>w/o MCP version</h3>
<p>This project demonstrates how to combine large reasoning models (LRMs) and CV-based model to finish reliable diagnosis in chest X-ray vision.</p>
<h4>Repository Structure</h4>
<ul>
<li><p><strong>Model_input</strong><br>Input classification results and location information.</p>
</li>
<li><p><strong>inference.py</strong><br>Use LRMs to infer the lessions&#39; possibilities baesd on the model input.</p>
</li>
</ul>
<h4>Prerequisites</h4>
<ul>
<li><strong>Python</strong>: 3.11 or later with packages: <code>pip install requirement.txt</code></li>
<li><strong>APIKEY</strong>: Use your API-Key of your preferred platform.</li>
</ul>
<hr>
<h2>📝 Citing <a name="citing"></a></h2>
<pre><code>..
</code></pre>
<hr>
