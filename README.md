<img src="bitcoin.jpg" align="right" />

# BitCoinPrediction [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
> pip install is coming

### Install
BitCoinPrediction requires [Python 3.6](https://www.python.org/downloads/release/python-360/).
Install dependencies thanks to setup.py
```
$ python setup.py
```

### Run
```python
from bitcoinpred.models.train_model import TrainLSTM

path = os.path.join(converged_data_path, "merged.csv")
TrainLSTM(path).fit_transform()
```

### Todos
 - Write Tests
 - Make pip install
 - ...

License
----
MIT

**Free Software !**
