import asyncio
import json
import binascii

async def test_stratum_duplex():
    reader, writer = await asyncio.open_connection('rvn.2miners.com', 6060)
    user = 'RAvjtY1eZxNB3aXe55CQQtKGfezgDXXFuP.grok_miner'

    # Subscribe
    writer.write(json.dumps({'id': 1, 'method': 'mining.subscribe', 'params': [user, 'x']}).encode() + b'\n')
    await writer.drain()
    sub_resp = json.loads(await reader.readline())
    print("Subscribe:", sub_resp)

    # Authorize
    writer.write(json.dumps({'id': 2, 'method': 'mining.authorize', 'params': [user, 'x']}).encode() + b'\n')
    await writer.drain()
    auth_resp = json.loads(await reader.readline())
    print("Authorize:", auth_resp)

    # Listen for 3 messages from pool
    for i in range(3):
        line = await reader.readline()
        if not line:
            break
        msg = json.loads(line)
        print(f"Message {i+1} from pool:", msg)

    writer.close()
    await writer.wait_closed()

asyncio.run(test_stratum_duplex())
