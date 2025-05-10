# client_proxy.py
import asyncio
import sys
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

async def main(image_path: str):
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["torch_mcp_server.py"],  
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "predict_via_flask",           
                arguments={"image_path": image_path}
            )
            print("Inference result:", result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_path>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
