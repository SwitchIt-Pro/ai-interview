"""
Conversational AI Scout - Voice Interview System API Server
Optimized for RTX 1650 (4GB VRAM)
"""

import asyncio
import json
import logging
import websockets
from pipeline.orchestrator import InterviewOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

connected_clients = set()
orchestrator = None
orchestrator_task = None

async def broadcast_event(event: dict):
    if not connected_clients:
        return
    message = json.dumps(event)
    for ws in list(connected_clients):
        try:
            await ws.send(message)
        except Exception as e:
            logger.error(f"Failed to send to a client: {e}")
            connected_clients.discard(ws)

async def handle_client(websocket):
    global orchestrator, orchestrator_task
    connected_clients.add(websocket)
    logger.info("New frontend client connected.")
    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get("command")
            if cmd == "start":
                if orchestrator and orchestrator._is_running:
                    logger.info("Interview already running.")
                    continue
                logger.info("Received start command from frontend.")
                orchestrator = InterviewOrchestrator(event_callback=broadcast_event)
                orchestrator_task = asyncio.create_task(orchestrator.start())
            elif cmd == "stop":
                if orchestrator and orchestrator._is_running:
                    logger.info("Received stop command from frontend.")
                    orchestrator.stop()
                    if orchestrator_task:
                        orchestrator_task.cancel()
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connected_clients.discard(websocket)
        logger.info("Frontend client disconnected.")

async def main():
    logger.info("Starting WebSocket Server for AI Scout on ws://localhost:8765 ...")
    logger.info("GPU: RTX 1650 (4GB VRAM) - Low-VRAM optimized mode active")
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
