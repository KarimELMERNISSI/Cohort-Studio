# License Dependencies and Compliance

## Overview

This document analyzes the open-source licenses of the libraries used in the Cohort Monitoring Package to determine the licensing constraints for the application itself.

## License Summary

All production dependencies use **Permissive** or **Weak Copyleft** licenses (MIT, BSD, Apache 2.0, MPL 2.0). There are **no GPL-licensed dependencies** in the project.

### License Types Found

- **Permissive**: MIT, BSD-3-Clause, Apache 2.0 (e.g., `streamlit`, `pandas`, `scikit-learn`, `numpy`, `RapidFuzz`)
- **Weak Copyleft**: MPL 2.0 (e.g., `certifi`, `tqdm`)
- **Proprietary (Free)**: `yfiles-graphs-for-streamlit` (free, non-transferable yWorks license)

> [!NOTE]
> Previous versions of this document flagged `Unidecode` (GPL) and `python-Levenshtein` (GPL-2.0) as risks. Both have been removed — `Unidecode` was dropped, and `python-Levenshtein` was replaced by `RapidFuzz` (MIT).

## Recommended Application License

Since all dependencies are permissively licensed, the project can be released under any license:

- **MIT License** — maximally permissive, simple
- **Apache License 2.0** — includes patent grant
- **Proprietary** — also compatible

No GPL-based restrictions apply.

## Detailed License List

The following table lists the licenses for all direct and transitive dependencies found in `requirements.txt`.

| Package | License |
| :--- | :--- |
| **aiobotocore** | Apache-2.0 |
| **aiohappyeyeballs** | PSF-2.0 |
| **aiohttp** | Apache-2.0 |
| **aioitertools** | MIT |
| **aiosignal** | Apache-2.0 |
| **alabaster** | BSD |
| **altair** | BSD-3-Clause |
| **annotated-doc** | MIT |
| **annotated-types** | MIT |
| **anyio** | MIT |
| **appdirs** | MIT |
| **archspec** | Apache-2.0 OR MIT |
| **argon2-cffi** | MIT |
| **argon2-cffi-bindings** | MIT |
| **arrow** | Apache-2.0 |
| **asgiref** | BSD-3-Clause |
| **astropy** | BSD-3-Clause |
| **astropy-iers-data** | BSD-3-Clause |
| **asttokens** | Apache-2.0 |
| **async-lru** | MIT |
| **atomicwrites** | MIT |
| **attrs** | MIT |
| **Automat** | MIT |
| **Babel** | BSD-3-Clause |
| **backoff** | MIT |
| **backports.functools-lru-cache** | PSF |
| **backports.tempfile** | PSF |
| **backports.weakref** | PSF |
| **bcrypt** | Apache-2.0 |
| **beautifulsoup4** | MIT |
| **binaryornot** | BSD-3-Clause |
| **black** | MIT |
| **bleach** | Apache-2.0 |
| **blinker** | MIT |
| **bokeh** | BSD-3-Clause |
| **boltons** | BSD-3-Clause |
| **botocore** | Apache-2.0 |
| **Bottleneck** | BSD-2-Clause |
| **Brotli** | MIT |
| **build** | MIT |
| **cachetools** | MIT |
| **certifi** | MPL-2.0 |
| **cffi** | MIT |
| **chardet** | LGPL-2.1 |
| **charset-normalizer** | MIT |
| **chroma-hnswlib** | Apache-2.0 |
| **click** | BSD-3-Clause |
| **cloudpickle** | BSD-3-Clause |
| **colorama** | BSD-3-Clause |
| **colorcet** | CC-BY |
| **coloredlogs** | MIT |
| **comm** | BSD-3-Clause |
| **constantly** | MIT |
| **contourpy** | BSD-3-Clause |
| **cryptography** | Apache-2.0 OR BSD-3-Clause |
| **cssselect** | BSD-3-Clause |
| **cycler** | BSD-3-Clause |
| **cytoolz** | BSD-3-Clause |
| **dask** | BSD-3-Clause |
| **dask-expr** | BSD-3-Clause |
| **dataclasses-json** | MIT |
| **datashader** | BSD-3-Clause |
| **debugpy** | MIT |
| **decorator** | BSD-2-Clause |
| **defusedxml** | PSF |
| **detect-delimiter** | Apache-2.0 |
| **diff-match-patch** | Apache-2.0 |
| **dill** | BSD-3-Clause |
| **distributed** | BSD-3-Clause |
| **distro** | Apache-2.0 |
| **docstring-to-markdown** | LGPL-2.1-or-later |
| **docutils** | BSD-2-Clause / PSF / Public Domain |
| **duckdb** | MIT |
| **entrypoints** | MIT |
| **et-xmlfile** | MIT |
| **executing** | MIT |
| **fastapi** | MIT |
| **fastjsonschema** | BSD-3-Clause |
| **filelock** | Unlicense |
| **filetype** | MIT |
| **Flask** | BSD-3-Clause |
| **fonttools** | MIT |
| **frozendict** | LGPL-3.0 |
| **frozenlist** | Apache-2.0 |
| **fsspec** | BSD-3-Clause |
| **future** | MIT |
| **gitdb** | BSD-3-Clause |
| **GitPython** | BSD-3-Clause |
| **google-api-core** | Apache-2.0 |
| **google-api-python-client** | Apache-2.0 |
| **google-auth** | Apache-2.0 |
| **google-auth-httplib2** | Apache-2.0 |
| **google-genai** | Apache-2.0 |
| **googleapis-common-protos** | Apache-2.0 |
| **greenlet** | MIT |
| **groq** | Apache-2.0 |
| **grpcio** | Apache-2.0 |
| **h11** | MIT |
| **h5py** | BSD-3-Clause |
| **HeapDict** | BSD-3-Clause |
| **hf-xet** | Apache-2.0 |
| **holoviews** | BSD-3-Clause |
| **httpcore** | BSD-3-Clause |
| **httplib2** | MIT |
| **httptools** | MIT |
| **httpx** | BSD-3-Clause |
| **httpx-sse** | MIT |
| **huggingface-hub** | Apache-2.0 |
| **humanfriendly** | MIT |
| **hvplot** | BSD-3-Clause |
| **hyperlink** | MIT |
| **idna** | BSD-3-Clause |
| **imagecodecs** | BSD-3-Clause |
| **imageio** | BSD-2-Clause |
| **imagesize** | MIT |
| **imbalanced-learn** | MIT |
| **importlib-metadata** | Apache-2.0 |
| **importlib_resources** | Apache-2.0 |
| **incremental** | MIT |
| **inflection** | MIT |
| **iniconfig** | MIT |
| **intake** | BSD-3-Clause |
| **intervaltree** | Apache-2.0 |
| **ipykernel** | BSD-3-Clause |
| **ipython** | BSD-3-Clause |
| **ipython-genutils** | BSD-3-Clause |
| **ipywidgets** | BSD-3-Clause |
| **isort** | MIT |
| **itemadapter** | BSD-3-Clause |
| **itemloaders** | BSD-3-Clause |
| **itsdangerous** | BSD-3-Clause |
| **jaraco.classes** | MIT |
| **jedi** | MIT |
| **jellyfish** | MIT |
| **Jinja2** | BSD-3-Clause |
| **jmespath** | MIT |
| **joblib** | BSD-3-Clause |
| **json5** | Apache-2.0 |
| **jsonpatch** | BSD-3-Clause |
| **jsonpointer** | BSD-3-Clause |
| **jsonschema** | MIT |
| **jsonschema-specifications** | MIT |
| **keyring** | MIT |
| **kiwisolver** | BSD-3-Clause |
| **kubernetes** | Apache-2.0 |
| **langchain** | MIT |
| **langchain-chroma** | MIT |
| **langchain-classic** | MIT |
| **langchain-community** | MIT |
| **langchain-core** | MIT |
| **langchain-google-genai** | MIT |
| **langchain-text-splitters** | MIT |
| **langgraph** | MIT |
| **langgraph-checkpoint** | MIT |
| **langgraph-prebuilt** | MIT |
| **langgraph-sdk** | MIT |
| **langsmith** | MIT |
| **lazy-object-proxy** | BSD-2-Clause |
| **lazy_loader** | BSD-3-Clause |
| **libarchive-c** | CC0 |
| **lightgbm** | MIT |
| **linkify-it-py** | MIT |
| **llvmlite** | BSD-2-Clause |
| **lmdb** | OLDAP-2.8 |
| **locket** | BSD-2-Clause |
| **lxml** | BSD-3-Clause |
| **lz4** | BSD-3-Clause |
| **Markdown** | BSD-3-Clause |
| **markdown-it-py** | MIT |
| **MarkupSafe** | BSD-3-Clause |
| **marshmallow** | MIT |
| **matplotlib** | PSF |
| **matplotlib-inline** | BSD-3-Clause |
| **mccabe** | MIT (Expat) |
| **mdit-py-plugins** | MIT |
| **mdurl** | MIT |
| **mistune** | BSD-3-Clause |
| **mmh3** | MIT |
| **more-itertools** | MIT |
| **mpmath** | BSD-3-Clause |
| **msgpack** | Apache-2.0 |
| **multidict** | Apache-2.0 |
| **multipledispatch** | BSD-3-Clause |
| **munkres** | Apache-2.0 |
| **mypy** | MIT |
| **mypy_extensions** | MIT |
| **narwhals** | MIT |
| **nest-asyncio** | BSD-2-Clause |
| **networkx** | BSD-3-Clause |
| **nltk** | Apache-2.0 |
| **numba** | BSD-2-Clause |
| **numexpr** | MIT |
| **numpy** | BSD-3-Clause |
| **numpydoc** | BSD-3-Clause |
| **oauthlib** | BSD-3-Clause |
| **onnxruntime** | MIT |
| **openpyxl** | MIT |
| **opentelemetry-instrumentation** | Apache-2.0 |
| **opentelemetry-instrumentation-asgi** | Apache-2.0 |
| **opentelemetry-instrumentation-fastapi** | Apache-2.0 |
| **opentelemetry-util-http** | Apache-2.0 |
| **orjson** | Apache-2.0 OR MIT |
| **ormsgpack** | Apache-2.0 OR MIT |
| **overrides** | Apache-2.0 |
| **packaging** | Apache-2.0 |
| **pandas** | BSD-3-Clause |
| **pandocfilters** | BSD-3-Clause |
| **panel** | BSD-3-Clause |
| **param** | BSD-3-Clause |
| **paramiko** | LGPL-2.1 |
| **parsel** | BSD-3-Clause |
| **parso** | MIT |
| **partd** | BSD-3-Clause |
| **pathlib** | MIT |
| **pathspec** | MPL-2.0 |
| **patsy** | BSD-2-Clause |
| **pexpect** | ISC |
| **pickleshare** | MIT |
| **pillow** | HPND |
| **pkce** | MIT |
| **pkginfo** | MIT |
| **platformdirs** | MIT |
| **plotly** | MIT |
| **pluggy** | MIT |
| **ply** | BSD-3-Clause |
| **posthog** | MIT |
| **prince** | MIT |
| **prometheus-client** | Apache-2.0 |
| **prompt-toolkit** | BSD-3-Clause |
| **propcache** | Apache-2.0 |
| **Protego** | BSD-3-Clause |
| **proto-plus** | Apache-2.0 |
| **protobuf** | BSD-3-Clause |
| **psutil** | BSD-3-Clause |
| **ptyprocess** | ISC |
| **pure-eval** | MIT |
| **py-cpuinfo** | MIT |
| **pyarrow** | Apache-2.0 |
| **pyasn1** | BSD-2-Clause |
| **pyasn1_modules** | BSD-2-Clause |
| **pybase64** | BSD-2-Clause |
| **pycodestyle** | MIT |
| **pycparser** | BSD-3-Clause |
| **pydantic** | MIT |
| **pydantic-settings** | MIT |
| **pydantic_core** | MIT |
| **pydeck** | Apache-2.0 |
| **PyDispatcher** | BSD-3-Clause |
| **pydocstyle** | MIT |
| **pyerfa** | BSD-3-Clause |
| **Pygments** | BSD-2-Clause |
| **PyJWT** | MIT |
| **PyNaCl** | Apache-2.0 |
| **pynndescent** | BSD-2-Clause |
| **pyodbc** | MIT |
| **pyOpenSSL** | Apache-2.0 |
| **pyparsing** | MIT |
| **pypdf** | BSD-3-Clause |
| **pyproject_hooks** | MIT |
| **PySocks** | BSD-3-Clause |
| **python-dateutil** | Apache-2.0 / BSD-3-Clause (Dual) |
| **python-docx** | MIT |
| **python-dotenv** | BSD-3-Clause |
| **python-json-logger** | BSD-2-Clause |
| **python-snappy** | BSD-3-Clause |
| **pytoolconfig** | LGPL-3.0-or-later |
| **pytz** | MIT |
| **pyviz_comms** | BSD-3-Clause |
| **pywavelets** | MIT |
| **PyYAML** | MIT |
| **pyzmq** | BSD-3-Clause / LGPL-3.0 |
| **qstylizer** | MIT |
| **queuelib** | BSD-3-Clause |
| **RapidFuzz** | MIT |
| **rdflib** | BSD-3-Clause |
| **readchar** | MIT |
| **referencing** | MIT |
| **regex** | Apache-2.0 |
| **requests** | Apache-2.0 |
| **requests-file** | Apache-2.0 |
| **requests-oauthlib** | ISC |
| **requests-toolbelt** | Apache-2.0 |
| **rfc3339-validator** | MIT |
| **rfc3986-validator** | MIT |
| **rich** | MIT |
| **rope** | LGPL-3.0-or-later |
| **rpds-py** | MIT |
| **rsa** | Apache-2.0 |
| **Rtree** | MIT |
| **ruamel.yaml** | MIT |
| **s3fs** | BSD-3-Clause |
| **scikit-image** | BSD-3-Clause |
| **scikit-learn** | BSD-3-Clause |
| **scipy** | BSD-3-Clause |
| **Scrapy** | BSD-3-Clause |
| **seaborn** | BSD-3-Clause |
| **semver** | BSD-3-Clause |
| **Send2Trash** | BSD-3-Clause |
| **sentence-transformers** | Apache-2.0 |
| **service-identity** | MIT |
| **setuptools** | MIT |
| **shellingham** | ISC |
| **sip** | SIP (Permissive) |
| **six** | MIT |
| **smart-open** | MIT |
| **smmap** | BSD-3-Clause |
| **sniffio** | MIT OR Apache-2.0 |
| **snowballstemmer** | BSD-3-Clause |
| **sortedcontainers** | Apache-2.0 |
| **soupsieve** | MIT |
| **Sphinx** | BSD-2-Clause |
| **sphinxcontrib-applehelp** | BSD-2-Clause |
| **sphinxcontrib-devhelp** | BSD-2-Clause |
| **sphinxcontrib-htmlhelp** | BSD-2-Clause |
| **sphinxcontrib-jsmath** | BSD-2-Clause |
| **sphinxcontrib-serializinghtml** | BSD-2-Clause |
| **SQLAlchemy** | MIT |
| **stack-data** | MIT |
| **starlette** | BSD-3-Clause |
| **statsmodels** | BSD-3-Clause |
| **streamlit** | Apache-2.0 |
| **streamlit-agraph** | MIT |
| **sympy** | BSD-3-Clause |
| **tables** | BSD-3-Clause |
| **tabulate** | MIT |
| **tblib** | BSD-2-Clause |
| **tenacity** | Apache-2.0 |
| **textdistance** | MIT |
| **threadpoolctl** | BSD-3-Clause |
| **three-merge** | MIT |
| **tifffile** | BSD-3-Clause |
| **tinycss2** | BSD-3-Clause |
| **tldextract** | BSD-3-Clause |
| **toml** | MIT |
| **tomli** | MIT |
| **tomlkit** | MIT |
| **toolz** | BSD-3-Clause |
| **tornado** | Apache-2.0 |
| **tqdm** | MPL-2.0 AND MIT |
| **traitlets** | BSD-3-Clause |
| **truststore** | MIT |
| **Twisted** | MIT |
| **typer** | MIT |
| **typer-slim** | MIT |
| **typing-inspect** | MIT |
| **typing-inspection** | MIT |
| **typing_extensions** | PSF |
| **tzdata** | Apache-2.0 |
| **tzlocal** | MIT |
| **uc-micro-py** | MIT |
| **ujson** | BSD-3-Clause |
| **umap-learn** | BSD-3-Clause |
| **uritemplate** | BSD-3-Clause OR Apache-2.0 |
| **urllib3** | MIT |
| **uvicorn** | BSD-3-Clause |
| **validators** | MIT |
| **w3lib** | BSD-3-Clause |
| **watchdog** | Apache-2.0 |
| **watchfiles** | MIT |
| **wcwidth** | MIT |
| **webencodings** | BSD-3-Clause |
| **websocket-client** | Apache-2.0 |
| **websockets** | BSD-3-Clause |
| **Werkzeug** | BSD-3-Clause |
| **whatthepatch** | MIT |
| **wheel** | MIT |
| **win-inet-pton** | Public Domain |
| **wrapt** | BSD-2-Clause |
| **xarray** | Apache-2.0 |
| **xlwings** | BSD-3-Clause |
| **xxhash** | BSD-2-Clause |
| **xyzservices** | BSD-3-Clause |
| **yapf** | Apache-2.0 |
| **yarl** | Apache-2.0 |
| **yfiles-graphs-for-streamlit** | Proprietary (free, non-transferable) |
| **zict** | BSD-3-Clause |
| **zipp** | MIT |
| **zope.interface** | ZPL-2.1 |
| **zstandard** | BSD-3-Clause |
