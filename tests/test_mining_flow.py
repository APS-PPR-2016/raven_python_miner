import asyncio
import json
import binascii
import struct
import numpy as np

async def test_mining_flow():
    reader, writer = await asyncio.open_connection('rvn.2miners.com', 6060)
    user = 'RAvjtY1eZxNB3aXe55CQQtKGfezgDXXFuP.grok_miner'

    # Subscribe
    writer.write(json.dumps({'id': 1, 'method': 'mining.subscribe', 'params': [user, 'x']}).encode() + b'\n')
    await writer.drain()
    sub_resp = json.loads(await reader.readline())
    print("Subscribe response:", sub_resp)

    # Authorize
    writer.write(json.dumps({'id': 2, 'method': 'mining.authorize', 'params': [user, 'x']}).encode() + b'\n')
    await writer.drain()
    auth_resp = json.loads(await reader.readline())
    print("Authorize response:", auth_resp)

    # Read target and job
    for _ in range(2):
        line = await reader.readline()
        print("Incoming:", line.decode().strip())

    writer.close()
    await writer.wait_closed()

asyncio.run(test_mining_flow())
