"""Scan all .py files and print top-level imported module names (third-party only)."""
import os, re, sys

STDLIB = {
    'abc','ast','asyncio','base64','bisect','collections','concurrent','contextlib',
    'copy','csv','ctypes','dataclasses','datetime','decimal','difflib','email',
    'enum','errno','functools','gc','getpass','glob','gzip','hashlib','heapq',
    'hmac','html','http','importlib','inspect','io','itertools','json','logging',
    'math','mimetypes','multiprocessing','numbers','operator','os','pathlib',
    'pickle','platform','pprint','queue','random','re','secrets','shlex','shutil',
    'signal','site','socket','sqlite3','string','struct','subprocess','sys',
    'tempfile','textwrap','threading','time','timeit','tkinter','traceback',
    'types','typing','unicodedata','unittest','urllib','uuid','warnings','weakref',
    'xml','zipfile','zlib','posixpath','ntpath','stat','_thread','typing_extensions',
}

LOCAL = {
    'app_pages','manage','utils','enrich','explore','monitor','prompts',
    'config','data','tests','dev_scripts','images','documentation',
}

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
imports = set()

for dirpath, _, fnames in os.walk(root):
    if '__pycache__' in dirpath or 'environments' in dirpath or '.git' in dirpath:
        continue
    for f in fnames:
        if f.endswith('.py'):
            try:
                with open(os.path.join(dirpath, f), encoding='utf-8', errors='ignore') as fh:
                    for line in fh:
                        line = line.strip()
                        m = re.match(r'^(?:from|import)\s+(\w+)', line)
                        if m:
                            mod = m.group(1)
                            if mod not in STDLIB and mod not in LOCAL and not mod.startswith('_'):
                                imports.add(mod)
            except:
                pass

for i in sorted(imports):
    print(i)
