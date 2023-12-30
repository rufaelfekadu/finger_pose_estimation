## Getting Started:

To use this plugin you will need the following:

1. The latest Gemini Ultraleap Hand Tracking Software. You can get this [here][developer-site-tracking-software].
2. An Ultraleap Hand Tracking Camera - follow setup process [here][developer-site-setup-camera].
3. Correctly installed "Bluetooth Low Energy C# sample" Windows application
4. Follow one of the Installation workflows listed below.


## Installation:

This module makes use of a compiled module called `leapc_cffi`. We include some pre-compiled python objects with our
Gemini installation from 5.17 onwards. Supported versions can be found [here](#pre-compiled-module-support). If you 
have the matching python version and have installed Gemini into the default location you can follow the steps below:

```
# Create and activate an evironment
conda create -n fpe python=3.10
conda activate fpe
```

```
# Install dependancies
pip install -r requirements.txt
pip install -e leapc-python-api
python examples/tracking_event_example.py
```