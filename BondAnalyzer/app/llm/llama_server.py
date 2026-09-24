from __future__ import annotations
import json, subprocess, time, urllib.request
from pathlib import Path
class LocalLLM:
 def __init__(self, root, port=8765):
  self.root=Path(root); self.port=port; self.process=None
  self.server=self.root/'runtime'/'llama'/'llama-server.exe'; self.models=sorted((self.root/'models').glob('*.gguf'))
 @property
 def available(self): return self.server.is_file() and bool(self.models)
 @property
 def base_url(self): return f'http://127.0.0.1:{self.port}'
 def __enter__(self):
  if not self.available: raise RuntimeError('Local LLM is not installed')
  self.process=subprocess.Popen([str(self.server),'-m',str(self.models[0]),'--host','127.0.0.1','--port',str(self.port),'-c','8192'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  deadline=time.time()+45
  while time.time()<deadline:
   try:
    with urllib.request.urlopen(self.base_url+'/health',timeout=1) as r:
     if r.status==200:return self
   except OSError: time.sleep(.5)
  self.stop(); raise RuntimeError('llama-server did not become ready')
 def complete(self,prompt):
  data=json.dumps({'messages':[{'role':'user','content':prompt}],'temperature':0,'response_format':{'type':'json_object'}}).encode()
  req=urllib.request.Request(self.base_url+'/v1/chat/completions',data=data,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=180) as r: return json.loads(r.read())['choices'][0]['message']['content']
 def stop(self):
  if self.process and self.process.poll() is None:self.process.terminate();
  if self.process:
   try:self.process.wait(timeout=5)
   except subprocess.TimeoutExpired:self.process.kill()
 def __exit__(self,*args):self.stop()
