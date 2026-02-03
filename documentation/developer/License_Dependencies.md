# License Dependencies and Compliance

## Overview

This document analyzes the open-source licenses of the libraries used in the Cohort Monitoring Package to determine the licensing constraints for the application itself.

## License Summary

The application mostly uses libraries with **Permissive** licenses (MIT, BSD, Apache 2.0), which allow for broad usage including proprietary distribution. However, there are a few **Copyleft** (GPL) dependencies that impose stricter requirements.

### License Types Found

- **Permissive**: MIT, BSD-3-Clause, Apache 2.0 (e.g., `streamlit`, `pandas`, `scikit-learn`, `numpy`)
- **Weak Copyleft**: MPL 2.0, LGPL (e.g., `certifi`, `chardet`)
- **Strong Copyleft (GPL)**: GPLv3 (e.g., `PyQt5`)

## Critical Dependencies (Restrictive Licenses)

The following packages are licensed under **GPL**, which generally requires that any application using them also be released under the GPL if distributed:

| Package | License | Impact | Recommendation |
| :--- | :--- | :--- | :--- |
| **`Unidecode`** | GPL | Forces application to be GPL. | **Remove**. Replaced by `text-unidecode` (Artistic/Permissive) or similar. |

> [!NOTE]
> `PyQt5` (GPL) was previously detected but has been confirmed as a local development artifact and is not a production dependency.

## Recommended Application License

### Option A: Open Source (GPLv3)

If you intend to keep the current dependencies (e.g. `Unidecode`):

- **Recommended License**: **GNU General Public License v3 (GPLv3)**.
- **Why**: This ensures compatibility with the strict copyleft licenses of your dependencies.

### Option B: Permissive (MIT / Apache 2.0)

If you wish to release under a permissive license (or keep the code proprietary):

1. **Remove `Unidecode`**.

- **Recommended License**: **MIT License** or **Apache License 2.0**.

## Detailed License List

The following table lists the licenses for all direct dependencies found in `requirements.txt`.

| Package | License |
| :--- | :--- |
| **aiobotocore** | Apache License 2.0 |
| **aiohappyeyeballs** | PSF-2.0 |
| **aiohttp** | Apache 2 |
| **aioitertools** | UNKNOWN |
| **aiosignal** | Apache 2 |
| **alabaster** | UNKNOWN |
| **altair** | BSD License |
| **annotated-doc** | Unknown |
| **annotated-types** | MIT License |
| **anyio** | MIT |
| **appdirs** | MIT |
| **archspec** | Apache-2.0 OR MIT |
| **argon2-cffi** | MIT License |
| **argon2-cffi-bindings** | MIT |
| **arrow** | Apache 2.0 |
| **asgiref** | BSD-3-Clause |
| **astropy** | BSD-3-Clause |
| **astropy-iers-data** | Copyright (c) 2023, Astropy Developers  All rights |
| **asttokens** | Apache 2.0 |
| **async-lru** | MIT License |
| **atomicwrites** | MIT |
| **attrs** | MIT License |
| **Automat** | MIT |
| **Babel** | BSD |
| **backoff** | MIT |
| **backports.functools-lru-cache** | UNKNOWN |
| **backports.tempfile** | Python Software Foundation License |
| **backports.weakref** | Python Software Foundation License |
| **bcrypt** | Apache License, Version 2.0 |
| **beautifulsoup4** | MIT License |
| **binaryornot** | BSD |
| **black** | MIT |
| **bleach** | Apache Software License |
| **blinker** | MIT License |
| **bokeh** | Copyright (c) Anaconda, Inc., and Bokeh Contributo |
| **boltons** | BSD |
| **botocore** | Apache License 2.0 |
| **Bottleneck** | Simplified BSD |
| **Brotli** | MIT |
| **build** | Unknown |
| **cachetools** | MIT |
| **certifi** | MPL-2.0 |
| **cffi** | MIT |
| **chardet** | LGPL |
| **charset-normalizer** | MIT |
| **chroma-hnswlib** | Unknown |
| **click** | BSD-3-Clause |
| **cloudpickle** | BSD-3-Clause |
| **colorama** | BSD License |
| **colorcet** | CC-BY License |
| **coloredlogs** | MIT |
| **comm** | BSD 3-Clause License  Copyright (c) 2022, Jupyter |
| **constantly** | MIT |
| **contourpy** | BSD 3-Clause License  Copyright (c) 2021-2023, Con |
| **cryptography** | Apache-2.0 OR BSD-3-Clause |
| **cssselect** | BSD |
| **cycler** | BSD |
| **cytoolz** | BSD |
| **dask** | BSD-3-Clause |
| **dask-expr** | BSD |
| **dataclasses-json** | MIT |
| **datashader** | New BSD |
| **debugpy** | MIT |
| **decorator** | new BSD License |
| **defusedxml** | PSFL |
| **detect-delimiter** | UNKNOWN |
| **diff-match-patch** | Apache |
| **dill** | BSD-3-Clause |
| **distributed** | BSD-3-Clause |
| **distro** | Apache License, Version 2.0 |
| **docstring-to-markdown** | LGPL-2.1-or-later |
| **docutils** | public domain, Python, 2-Clause BSD, GPL 3 (see CO |
| **duckdb** | MIT License |
| **entrypoints** | MIT License |
| **et-xmlfile** | MIT |
| **executing** | MIT |
| **fastapi** | Unknown |
| **fastjsonschema** | BSD |
| **filelock** | The Unlicense (Unlicense) |
| **filetype** | MIT |
| **Flask** | BSD License |
| **fonttools** | MIT |
| **frozendict** | LGPL v3 |
| **frozenlist** | Apache 2 |
| **fsspec** | BSD |
| **future** | MIT |
| **gitdb** | BSD License |
| **GitPython** | BSD-3-Clause |
| **google-api-core** | Apache 2.0 |
| **google-api-python-client** | Apache 2.0 |
| **google-auth** | Apache 2.0 |
| **google-auth-httplib2** | Apache 2.0 |
| **google-genai** | Unknown |
| **googleapis-common-protos** | Apache 2.0 |
| **greenlet** | MIT License |
| **groq** | Apache-2.0 |
| **grpcio** | Apache License 2.0 |
| **h11** | MIT |
| **h5py** | BSD-3-Clause |
| **HeapDict** | BSD |
| **hf-xet** | Apache Software License |
| **holoviews** | BSD |
| **httpcore** | BSD License |
| **httplib2** | MIT |
| **httptools** | Unknown |
| **httpx** | BSD License |
| **httpx-sse** | MIT |
| **huggingface-hub** | Apache |
| **humanfriendly** | MIT |
| **hvplot** | BSD |
| **hyperlink** | MIT |
| **idna** | BSD License |
| **imagecodecs** | BSD |
| **imageio** | BSD-2-Clause |
| **imagesize** | MIT |
| **imbalanced-learn** | MIT |
| **importlib-metadata** | Apache Software License |
| **importlib_resources** | Apache Software License |
| **incremental** | MIT |
| **inflection** | MIT |
| **iniconfig** | MIT License |
| **intake** | BSD |
| **intervaltree** | Apache License, Version 2.0 |
| **ipykernel** | BSD 3-Clause License  Copyright (c) 2015, IPython |
| **ipython** | BSD-3-Clause |
| **ipython-genutils** | BSD |
| **ipywidgets** | BSD |
| **isort** | MIT |
| **itemadapter** | BSD |
| **itemloaders** | BSD |
| **itsdangerous** | BSD License |
| **jaraco.classes** | UNKNOWN |
| **jedi** | MIT |
| **jellyfish** | MIT |
| **Jinja2** | BSD License |
| **jmespath** | MIT |
| **joblib** | BSD 3-Clause |
| **json5** | Apache |
| **jsonpatch** | Modified BSD License |
| **jsonpointer** | Modified BSD License |
| **jsonschema** | MIT |
| **jsonschema-specifications** | MIT |
| **keyring** | MIT License |
| **kiwisolver** | =========================  The Kiwi licensing term |
| **kubernetes** | Apache License Version 2.0 |
| **langchain** | Not Installed |
| **langchain-chroma** | MIT |
| **langchain-classic** | MIT |
| **langchain-community** | Not Installed |
| **langchain-core** | Not Installed |
| **langchain-google-genai** | Not Installed |
| **langchain-text-splitters** | Not Installed |
| **langgraph** | Unknown |
| **langgraph-checkpoint** | Unknown |
| **langgraph-prebuilt** | Unknown |
| **langgraph-sdk** | Unknown |
| **langsmith** | MIT |
| **lazy-object-proxy** | BSD-2-Clause |
| **lazy_loader** | BSD 3-Clause License  Copyright (c) 2022--2023, Sc |
| **libarchive-c** | CC0 |
| **lightgbm** | The MIT License (MIT)  Copyright (c) Microsoft Cor |
| **linkify-it-py** | MIT |
| **llvmlite** | BSD |
| **lmdb** | OLDAP-2.8 |
| **locket** | BSD-2-Clause |
| **lxml** | BSD-3-Clause |
| **lz4** | BSD License |
| **Markdown** | BSD License |
| **markdown-it-py** | MIT License |
| **MarkupSafe** | BSD-3-Clause |
| **marshmallow** | MIT License |
| **matplotlib** | PSF |
| **matplotlib-inline** | BSD 3-Clause |
| **mccabe** | Expat license |
| **mdit-py-plugins** | MIT |
| **mdurl** | MIT License |
| **mistune** | BSD 3-Clause License |
| **mmh3** | MIT License  Copyright (c) 2011-2025 Hajime Senuma |
| **more-itertools** | MIT License |
| **mpmath** | BSD |
| **msgpack** | Apache 2.0 |
| **multidict** | Apache 2 |
| **multipledispatch** | BSD |
| **munkres** | Apache Software License |
| **mypy** | MIT |
| **mypy_extensions** | Unknown |
| **narwhals** | MIT License |
| **nest-asyncio** | BSD |
| **networkx** | BSD License |
| **nltk** | Apache License, Version 2.0 |
| **numba** | BSD |
| **numexpr** | MIT |
| **numpy** | Copyright (c) 2005-2023, NumPy Developers. All rig |
| **numpydoc** | BSD |
| **oauthlib** | BSD-3-Clause |
| **onnxruntime** | MIT License |
| **openpyxl** | MIT |
| **opentelemetry-instrumentation** | Apache Software License |
| **opentelemetry-instrumentation-asgi** | Apache Software License |
| **opentelemetry-instrumentation-fastapi** | Apache Software License |
| **opentelemetry-util-http** | Apache Software License |
| **orjson** | Apache-2.0 OR MIT |
| **ormsgpack** | Apache-2.0 OR MIT |
| **overrides** | Apache License, Version 2.0 |
| **packaging** | Apache Software License |
| **pandas** | BSD 3-Clause License  Copyright (c) 2008-2011, AQR |
| **pandocfilters** | BSD-3-Clause |
| **panel** | BSD |
| **param** | BSD-3-Clause |
| **paramiko** | LGPL |
| **parsel** | BSD |
| **parso** | MIT |
| **partd** | BSD |
| **pathlib** | MIT License |
| **pathspec** | MPL 2.0 |
| **patsy** | 2-clause BSD |
| **pexpect** | ISC license |
| **pickleshare** | MIT |
| **pillow** | HPND |
| **pkce** | MIT |
| **pkginfo** | MIT |
| **platformdirs** | MIT License |
| **plotly** | MIT |
| **pluggy** | MIT |
| **ply** | BSD |
| **posthog** | MIT |
| **prince** | Unknown |
| **prometheus-client** | Apache Software License 2.0 |
| **prompt-toolkit** | BSD License |
| **propcache** | Apache-2.0 |
| **Protego** | BSD |
| **proto-plus** | Apache 2.0 |
| **protobuf** | BSD-3-Clause |
| **psutil** | BSD |
| **ptyprocess** | UNKNOWN |
| **pure-eval** | MIT |
| **py-cpuinfo** | MIT |
| **pyarrow** | Apache License, Version 2.0 |
| **pyasn1** | BSD |
| **pyasn1_modules** | BSD |
| **pybase64** | BSD-2-Clause |
| **pycodestyle** | MIT |
| **pycparser** | BSD |
| **pydantic** | MIT License |
| **pydantic-settings** | MIT License |
| **pydantic_core** | MIT |
| **pydeck** | Apache License 2.0 |
| **PyDispatcher** | BSD |
| **pydocstyle** | MIT |
| **pyerfa** | BSD 3-Clause License |
| **Pygments** | BSD-2-Clause |
| **PyJWT** | MIT |
| **PyNaCl** | Apache License 2.0 |
| **pynndescent** | BSD |
| **pyodbc** | MIT License |
| **pyOpenSSL** | Apache License, Version 2.0 |
| **pyparsing** | MIT License |
| **pypdf** | Not Installed |
| **pyproject_hooks** | MIT License |
| **PySocks** | BSD |
| **python-dateutil** | Dual License |
| **python-docx** | MIT |
| **python-dotenv** | BSD-3-Clause |
| **python-json-logger** | BSD |
| **python-snappy** | BSD |
| **pytoolconfig** | LGPL-3.0-or-later |
| **pytz** | MIT |
| **pyviz_comms** | BSD 3-Clause License  Copyright (c) 2023, Philipp |
| **pywavelets** | Copyright (c) 2006-2012 Filip Wasilewski <<http://e> |
| **PyYAML** | MIT |
| **pyzmq** | LGPL+BSD |
| **qstylizer** | MIT |
| **queuelib** | BSD |
| **RapidFuzz** | Unknown |
| **rdflib** | BSD-3-Clause |
| **readchar** | MIT |
| **referencing** | MIT |
| **regex** | Apache Software License |
| **requests** | Apache-2.0 |
| **requests-file** | Apache 2.0 |
| **requests-oauthlib** | ISC |
| **requests-toolbelt** | Apache 2.0 |
| **rfc3339-validator** | MIT license |
| **rfc3986-validator** | MIT license |
| **rich** | MIT |
| **rope** | LGPL-3.0-or-later |
| **rpds-py** | MIT |
| **rsa** | Apache-2.0 |
| **Rtree** | MIT |
| **ruamel.yaml** | MIT license |
| **s3fs** | BSD |
| **scikit-image** | Files: * Copyright: 2009-2022 the scikit-image tea |
| **scikit-learn** | Unknown |
| **scipy** | Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, |
| **Scrapy** | BSD |
| **seaborn** | BSD License |
| **semver** | BSD |
| **Send2Trash** | BSD License |
| **sentence-transformers** | Not Installed |
| **service-identity** | MIT |
| **setuptools** | MIT License |
| **shellingham** | ISC License |
| **sip** | SIP |
| **six** | MIT |
| **smart-open** | MIT |
| **smmap** | BSD |
| **sniffio** | MIT OR Apache-2.0 |
| **snowballstemmer** | BSD-3-Clause |
| **sortedcontainers** | Apache 2.0 |
| **soupsieve** | MIT License |
| **Sphinx** | BSD |
| **sphinxcontrib-applehelp** | BSD |
| **sphinxcontrib-devhelp** | BSD |
| **sphinxcontrib-htmlhelp** | BSD |
| **sphinxcontrib-jsmath** | BSD |
| **sphinxcontrib-serializinghtml** | BSD |
| **SQLAlchemy** | MIT |
| **stack-data** | MIT |
| **starlette** | Unknown |
| **statsmodels** | BSD License |
| **streamlit** | Apache License 2.0 |
| **streamlit-agraph** | UNKNOWN |
| **sympy** | BSD |
| **tables** | BSD 3-Clause License |
| **tabulate** | MIT |
| **tblib** | BSD-2-Clause |
| **tenacity** | Apache 2.0 |
| **textdistance** | MIT |
| **threadpoolctl** | BSD-3-Clause |
| **three-merge** | MIT |
| **tifffile** | BSD |
| **tinycss2** | BSD License |
| **tldextract** | BSD License |
| **toml** | MIT |
| **tomli** | MIT License |
| **tomlkit** | Unknown |
| **toolz** | BSD |
| **tornado** | Apache-2.0 |
| **tqdm** | MPL-2.0 AND MIT |
| **traitlets** | BSD 3-Clause License  - Copyright (c) 2001-, IPyth |
| **truststore** | MIT License |
| **Twisted** | MIT License |
| **typer** | UNKNOWN |
| **typer-slim** | MIT License |
| **typing-inspect** | MIT |
| **typing-inspection** | Unknown |
| **typing_extensions** | Unknown |
| **tzdata** | Apache-2.0 |
| **tzlocal** | MIT |
| **uc-micro-py** | MIT |
| **ujson** | BSD License |
| **umap-learn** | BSD |
| **uritemplate** | BSD 3-Clause OR Apache-2.0 |
| **urllib3** | MIT License |
| **uvicorn** | Unknown |
| **validators** | MIT |
| **w3lib** | BSD |
| **watchdog** | Apache-2.0 |
| **watchfiles** | MIT |
| **wcwidth** | MIT |
| **webencodings** | BSD |
| **websocket-client** | Apache-2.0 |
| **websockets** | BSD-3-Clause |
| **Werkzeug** | BSD License |
| **whatthepatch** | MIT |
| **wheel** | MIT License |
| **win-inet-pton** | This software released into the public domain. Any |
| **wrapt** | BSD |
| **xarray** | Apache-2.0 |
| **xlwings** | BSD 3-clause |
| **xxhash** | BSD |
| **xyzservices** | 3-Clause BSD |
| **yapf** | Apache License |
| **yarl** | Apache-2.0 |
| **yfiles-graphs-for-streamlit** | Not Installed |
| **zict** | BSD |
| **zipp** | MIT License |
| **zope.interface** | ZPL 2.1 |
| **zstandard** | BSD |
