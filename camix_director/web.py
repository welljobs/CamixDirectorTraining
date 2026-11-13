"""Loopback-only, authenticated, dependency-free local web training console."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import secrets
import signal
import threading
from urllib.parse import parse_qs, urlparse
import wave
from http.cookies import SimpleCookie
import re
import webbrowser
import zipfile

import numpy as np
from .web_core import CODE_ROOT, JobManager, scoped, relative, read_json, pack_info

ASSETS=Path(__file__).with_name('web_assets')


def strict_json(raw):
    def invalid(value):raise ValueError('JSON不接受NaN/Infinity：'+value)
    return json.loads(raw,parse_constant=invalid)


def preview(root,payload,audio=False):
    path=scoped(root,payload.get('path',[''])[0],True)
    if not any(path.is_relative_to(Path(root)/folder) for folder in ('data','datasets/objects')) or path.suffix!='.npz':raise ValueError('仅预览 data 内的数据包')
    with zipfile.ZipFile(path) as archive:
        member='audio.npy' if audio else 'frames.npy'
        if archive.getinfo(member).file_size>128*1024*1024:raise ValueError('该数组展开超过128MiB，网页不直接加载预览；请准备更短片段')
    with np.load(path,allow_pickle=False) as values:
        array=values['audio' if audio else 'frames']
        tick=int(payload.get('tick',['0'])[0]);camera=int(payload.get('camera',['0'])[0])
        if not 0<=tick<len(array) or not 0<=camera<array.shape[1]:raise ValueError('预览时刻／机位超出范围')
        if audio:
            meta=json.loads(str(values['metadata'].item()));rate=int(meta['config']['sample_rate'])
            pcm=np.clip(array[tick,camera],-1,1);out=io.BytesIO()
            with wave.open(out,'wb') as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(rate);wav.writeframes((pcm*32767).astype('<i2').tobytes())
            return out.getvalue(),'audio/wav'
        from PIL import Image
        n=array.shape[1];canvas=Image.new('RGB',(n*240,240),'#101419')
        for index in range(n):canvas.paste(Image.fromarray(array[tick,index]).resize((240,240)),(index*240,0))
        out=io.BytesIO();canvas.save(out,format='JPEG',quality=88)
        return out.getvalue(),'image/jpeg'


def make_server(manager,port=8765,token=None):
    secret=token or secrets.token_urlsafe(32)
    from .studio_api import Studio
    studio=Studio(manager)
    class Handler(BaseHTTPRequestHandler):
        server_version='CamixLocal/0.2'
        def log_message(self,*args):pass  # URLs/tokens never go into access logs.
        def _security(self,api=False):
            host=self.headers.get('Host','')
            allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            if host not in allowed:raise PermissionError('Invalid Host; loopback access only')
            origin=self.headers.get('Origin')
            if origin and origin not in {f'http://{x}' for x in allowed}:raise PermissionError('Cross-origin requests are refused')
            # A normal top-level link may come from another site (e.g. a chat).
            # Only the public entry document is exempt; APIs, media and embeds
            # retain the cross-site block, in addition to Host/Origin/token checks.
            entry_navigation=(self.command=='GET' and urlparse(self.path).path in ('/','/index.html','/v2','/v2.html')
                and self.headers.get('Sec-Fetch-Mode')=='navigate'
                and self.headers.get('Sec-Fetch-Dest')=='document')
            if self.headers.get('Sec-Fetch-Site')=='cross-site' and not entry_navigation:
                raise PermissionError('Cross-site requests are refused')
            if api and not secrets.compare_digest(self.headers.get('X-Camix-Token',''),secret):raise PermissionError('缺少本地会话令牌；请使用启动终端显示的完整地址')
        def _headers(self,kind,length,status=200):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(length))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Referrer-Policy','no-referrer')
            if self.headers.get('X-Camix-Token')==secret:
                self.send_header('Set-Cookie',f'camix_media_{self.server.server_port}={secret}; HttpOnly; SameSite=Strict; Path=/; Max-Age=86400')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        def _bytes(self,data,kind,status=200):
            self._headers(kind,len(data),status);self.end_headers();self.wfile.write(data)
        def _json(self,payload,status=200):
            self._bytes(json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8'),'application/json; charset=utf-8',status)
        def _error(self,error):
            if isinstance(error,(BrokenPipeError,ConnectionResetError)):return
            code=403 if isinstance(error,PermissionError) else 400 if isinstance(error,(ValueError,TypeError,KeyError,FileNotFoundError)) else 500
            try:self._json(dict(error=str(error)),code)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            try:
                parsed=urlparse(self.path);route=parsed.path;query=parse_qs(parsed.query)
                self._security(route.startswith('/api/') and route!='/api/media')
                if route=='/api/media':return self._media(query)
                if route in ('/','/index.html','/app.js','/style.css','/studio.js','/studio.css','/v2','/v2.html','/v2.js','/v2.css'):
                    name='index.html' if route in ('/','/index.html') else 'v2.html' if route=='/v2' else route[1:]
                    path=ASSETS/name
                    return self._bytes(path.read_bytes(),{'index.html':'text/html; charset=utf-8','app.js':'text/javascript; charset=utf-8','style.css':'text/css; charset=utf-8','studio.js':'text/javascript; charset=utf-8','studio.css':'text/css; charset=utf-8','v2.html':'text/html; charset=utf-8','v2.js':'text/javascript; charset=utf-8','v2.css':'text/css; charset=utf-8'}[name])
                if route.startswith('/api/studio/'):return self._json(studio.get(route,query))
                if route=='/api/state':
                    import shutil
                    return self._json(dict(workspace=str(manager.root),system=manager.system,free_disk_bytes=shutil.disk_usage(manager.root).free,
                        catalog=manager.catalog(force=query.get('refresh')==['1']),jobs=manager.jobs()))
                if route=='/api/browse':return self._json(manager.browse(query.get('path',['data'])[0]))
                if route=='/api/logs':return self._json(dict(text=manager.logs(query.get('id',[''])[0])))
                if route=='/api/job':return self._json(manager.get(query.get('id',[''])[0]))
                if route in ('/api/preview','/api/audio'):
                    data,kind=preview(manager.root,query,route=='/api/audio');return self._bytes(data,kind)
                if route=='/api/artifact':
                    path=scoped(manager.root,query.get('path',[''])[0],True)
                    if not any(path.is_relative_to(manager.root/name) for name in ('runs','reports','data','datasets')) or not path.is_file() or path.suffix.lower() not in ('.pt','.json','.jsonl','.png','.jpg','.npz','.md'):
                        raise ValueError('该文件不是可下载的模型／报告／训练数据')
                    if path.stat().st_size>512*1024*1024:raise ValueError('网页下载限制512MiB；更大文件请从工作区直接复制')
                    self._headers('application/octet-stream',path.stat().st_size)
                    # No user-provided filename enters a response header.
                    self.send_header('Content-Disposition','attachment; filename="artifact'+path.suffix+'"');self.end_headers()
                    with path.open('rb') as source:
                        while block:=source.read(65536):self.wfile.write(block)
                    return
                self._json(dict(error='Not found'),404)
            except Exception as error:self._error(error)
        def _media(self,query,head=False):
            cookie=SimpleCookie();cookie.load(self.headers.get('Cookie',''))
            value=cookie.get(f'camix_media_{self.server.server_port}')
            if not value or not secrets.compare_digest(value.value,secret):raise PermissionError('Media session is not authenticated')
            path=scoped(manager.root,query.get('path',[''])[0],True)
            if not path.is_relative_to(manager.root/'data') or path.suffix.lower() not in ('.mp4','.m4v','.mov','.webm','.wav','.mp3','.ogg') or not path.is_file():raise ValueError('Only authorized data media can be streamed')
            size=path.stat().st_size;start=0;end=size-1;status=200
            range_header=self.headers.get('Range')
            if range_header:
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',range_header)
                if not match:raise ValueError('Only one byte range is accepted')
                left,right=match.groups()
                if not left and not right:raise ValueError('Empty byte range')
                if left:start=int(left);end=min(size-1,int(right)) if right else size-1
                else:start=max(0,size-int(right))
                if start>=size or end<start:
                    self._headers('application/octet-stream',0,416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
                status=206
            self._headers(mimetypes.guess_type(path.name)[0] or 'application/octet-stream',max(0,end-start+1),status)
            self.send_header('Accept-Ranges','bytes')
            if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            if head:return
            with path.open('rb') as source:
                source.seek(start);remaining=end-start+1
                while remaining:
                    block=source.read(min(65536,remaining))
                    if not block:break
                    self.wfile.write(block);remaining-=len(block)
        def do_HEAD(self):
            try:
                parsed=urlparse(self.path);self._security(False)
                if parsed.path!='/api/media':return self._bytes(b'','text/plain',404)
                self._media(parse_qs(parsed.query),head=True)
            except Exception as error:self._error(error)
        def do_POST(self):
            try:
                self._security(True)
                if self.headers.get('Content-Type','').split(';')[0]!='application/json':raise ValueError('Content-Type must be application/json')
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1024*1024:raise ValueError('请求大小必须在1MiB以内')
                payload=strict_json(self.rfile.read(length).decode('utf-8'))
                route=urlparse(self.path).path
                if route.startswith('/api/studio/'):
                    result=studio.post(route,payload);manager.catalog_at=0
                    return self._json(result)
                if route=='/api/jobs':return self._json(manager.start(payload),201)
                if route=='/api/stop':return self._json(manager.stop(payload.get('id')))
                self._json(dict(error='Not found'),404)
            except Exception as error:self._error(error)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    server.session_token=secret
    return server


def main():
    parser=argparse.ArgumentParser(description='Local training console: loopback only, no Camix app changes')
    parser.add_argument('--workspace',default=str(CODE_ROOT));parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    root=Path(args.workspace).expanduser().resolve()
    if not root.is_dir():parser.error('Workspace must be an existing directory')
    def stop_signal(signum,frame):raise KeyboardInterrupt
    for name in ("SIGTERM","SIGHUP","SIGBREAK"):
        if hasattr(signal,name):signal.signal(getattr(signal,name),stop_signal)
    manager=JobManager(root)
    try:server=make_server(manager,args.port)
    except OSError as error:
        raise SystemExit(f'本地端口不可用：{error}；可使用 --port 8766。不会自动结束其他程序。')
    url=f'http://127.0.0.1:{server.server_port}/#token={server.session_token}'
    print('\nCamix 本地网页训练台（仅本机访问）\n'+url+'\n请用Ctrl+C正常停止服务。实验模型不可接管真实节目。\n',flush=True)
    if not args.no_browser:threading.Timer(.5,lambda:webbrowser.open(url)).start()
    try:server.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:print('\n正在停止本地训练任务…',flush=True)
    finally:manager.close();server.server_close()


if __name__=='__main__':main()
