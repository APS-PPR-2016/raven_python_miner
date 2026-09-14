import socket, json
s = socket.create_connection(('rvn.2miners.com', 6060), timeout=10)
f = s.makefile('rw', encoding='utf-8')
# Subscribe
f.write(json.dumps({'id': 1, 'method': 'mining.subscribe', 'params': ['RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi.grok_miner', 'x']}) + '\n')
f.flush()
print('Sub response:', f.readline().strip())
# Authorize
f.write(json.dumps({'id': 2, 'method': 'mining.authorize', 'params': ['RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi.grok_miner', 'x']}) + '\n')
f.flush()
print('Auth response:', f.readline().strip())
for _ in range(4):
    line = f.readline()
    if not line: break
    msg = json.loads(line)
    print('Method/Msg:', msg.get('method'), msg.get('params'))
s.close()


# inspect notify params.
# .\.venv\Scripts\python.exe -c "code below"
import socket, json
try:
    ip = socket.gethostbyname('rvn.2miners.com')
    print('Resolved IP:', ip)
    s = socket.create_connection((ip, 6060), timeout=10)
    f = s.makefile('rw', encoding='utf-8')
    f.write(json.dumps({'id': 1, 'method': 'mining.subscribe', 'params': ['RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi.grok_miner', 'x']}) + '\n')
    f.flush()
    print('Sub response:', f.readline().strip())
    f.write(json.dumps({'id': 2, 'method': 'mining.authorize', 'params': ['RXWGbpXEaeD5vrzYLZQGEAgkWqHPtpvimi.grok_miner', 'x']}) + '\n')
    f.flush()
    print('Auth response:', f.readline().strip())
    for i in range(5):
        line = f.readline()
        if not line: break
        msg = json.loads(line)
        m = msg.get('method')
        p = msg.get('params')
        print('Msg', i, 'method:', m)
        if p:
            for idx, item in enumerate(p):
                print(f'  param[{idx}] = {item}')
    s.close()
except Exception as e:
    print('Exception:', e)
