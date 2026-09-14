import asyncio
import json

async def test_submit():
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

    # Read set_target & notify
    job_id = None
    header = None
    for _ in range(2):
        line = await reader.readline()
        msg = json.loads(line)
        print("Msg:", msg)
        if msg.get('method') == 'mining.notify':
            job_id = msg['params'][0]
            header = msg['params'][1]

    # Test submitting a share
    # Format for Kawpow stratum:
    # ["user", "job_id", "nonce" (hex 16 chars or 8 chars?), "header_hash", "mix_hash"]
    # Let's see how 2miners responds to different submit formats
    submit_msg = {
        "id": 10,
        "method": "mining.submit",
        "params": [user, job_id, "000000000000441f", header, "00" * 32]
    }
    print("Sending submit:", submit_msg)
    writer.write(json.dumps(submit_msg).encode() + b'\n')
    await writer.drain()

    # Read response
    resp = await reader.readline()
    print("Submit response:", resp.decode().strip())

    writer.close()
    await writer.wait_closed()

asyncio.run(test_submit())
