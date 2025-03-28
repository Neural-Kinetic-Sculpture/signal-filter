import asyncio
import websockets
import json

async def send_data(websocket, path):
    while True:
        filtered_data = {"alpha": 0.75, "beta": 0.25}
        await websocket.send(json.dumps(filtered_data))
        await asyncio.sleep(0.1)  # Send every 100ms

async def main():
    async with websockets.serve(send_data, "localhost", 8765):
        await asyncio.Future()  # Keep running forever

asyncio.run(main())
