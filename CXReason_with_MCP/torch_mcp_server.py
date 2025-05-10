from mcp.server.fastmcp import FastMCP
import requests
import os


mcp = FastMCP("XRayFlaskProxy")


@mcp.tool()
def predict_via_flask(image_path: str) -> list[dict]:
    # Ensure file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    url = "http://172.22.162.50:5050/predict"
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/png")}
        resp = requests.post(url, files=files)
    resp.raise_for_status()
    payload = resp.json()
    results = payload.get("results")
    if results is None:
        raise ValueError(f"Unexpected response format: {payload}")
    return results

if __name__ == "__main__":
    mcp.run()
