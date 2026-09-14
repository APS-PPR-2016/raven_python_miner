# .\.venv\Scripts\python.exe -c "code below"
import socket, json
ip = socket.gethostbyname('rvn.2miners.com')
s = socket.create_connection((ip, 6060), timeout=5)
f = s.makefile('rw', encoding='utf-8')
f.write(json.dumps({'id': 1, 'method': 'mining.subscribe', 'params': ['RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi.grok_miner', 'x']}) + '\n')
f.flush()
sub_res = json.loads(f.readline())
print('Sub res:', sub_res)
f.write(json.dumps({'id': 2, 'method': 'mining.authorize', 'params': ['RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi.grok_miner', 'x']}) + '\n')
f.flush()
auth_res = json.loads(f.readline())
print('Auth res:', auth_res)
line = f.readline()
notify = json.loads(f.readline())
print('Notify:', notify['params'][0])
job_id = notify['params'][0]
header_hash = notify['params'][1]

# Try sending different submit formats
# Format A: [worker, job_id, nonce, header_hash, mix_hash]
f.write(json.dumps({'id': 3, 'method': 'mining.submit', 'params': ['RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi.grok_miner', job_id, '0000000000000001', header_hash, '0'*64]}) + '\n')
f.flush()
print('Format A response:', f.readline().strip())

# Format B: [worker, job_id, nonce, mix_hash]
f.write(json.dumps({'id': 4, 'method': 'mining.submit', 'params': ['RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi.grok_miner', job_id, '0000000000000001', '0'*64]}) + '\n')
f.flush()
print('Format B response:', f.readline().strip())

# Format C: [worker, job_id, nonce]
f.write(json.dumps({'id': 5, 'method': 'mining.submit', 'params': ['RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi.grok_miner', job_id, '0000000000000001']}) + '\n')
f.flush()
print('Format C response:', f.readline().strip())
s.close()
