<img src="bitcoin.jpg" align="right" />

# BitCoinPrediction [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
> pip install is coming

### Installation
BitCoinPrediction requires [Python 3.6](https://www.python.org/downloads/release/python-360/).
Install dependencies thanks to setup.py
```python
$ python setup.py
```

### Run
```python
from mozinor.baboulinet import Baboulinet

cls = Baboulinet(filepath="toto.csv", y_col="predict", regression=False)
res = cls.babouline()
```

### Todos
 - Write Tests
 - Make pip install
 - ...

License
----
MIT

**Free Software !**
