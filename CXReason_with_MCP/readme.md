# Chest X-Ray Inference Service

## Overview

This project demonstrates a complete workflow for deploying a pre-trained DenseNet121 model (trained on chest X-ray images) as a Flask HTTP API, exposing it via an MCP server, and interacting through a Python client.

## Repository Structure

- **flask_torchxray.py**  
  Flask application wrapping the DenseNet121 model. Exposes a `/predict` POST endpoint that accepts an X-ray image and returns multi-label pathology probabilities in JSON.

- **torch_mcp_server.py**  
  MCP server implementation that registers a `predict_via_flask` tool, forwarding image inference requests to the Flask API.

- **client.py**  
  Python client demonstrating how to call the MCP server’s `predict_via_flask` tool and display the returned JSON results.

## Prerequisites

- **Python**: 3.8 or later  
- **Hardware**: (Optional) CUDA-enabled GPU for accelerated inference  
- **Tools**:  
  - `tmux` (recommended for long-running processes)  
  - MCP packages (`mcp-server`, `mcp-client`)
