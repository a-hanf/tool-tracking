# Tool Tracking Dataset - Supplement Code

In order to get things running:

1. Clone the repository

```
git clone https://github.com/mutschcr/tool-tracking.git
cd tool-tracking
```

2. Then you need to download the measurement data from an external host:
```
wget https://owncloud.fraunhofer.de/index.php/s/MQUpf2vhIghAtke/download -O tool-tracking-data.zip
unzip tool-tracking-data.zip && rm tool-tracking-data.zip
```

3. Setup a virtual python environment (e.g. with [conda](https://www.anaconda.com/))
```
conda create --name tool-tracking_env python=3.7
conda activate tool-tracking_env
pip install -r requirements.txt
```
4. Mark the data-tools and fhg-utils folders as Content roots or add them to your PYTHONPATH
```
export PYTHONPATH=$PYTHONPATH:fhg-utils/:data-tools/
```
5. Get introduced on how to load and work with the data

Start [Jupyter](https://jupyter.org/) and run the both notebook `How-to-load-the-data.ipynb` and `plot_window_sizes.ipynb` with:
```
jupyter notebook
```

# Best Results
e.g. query:
tags."mlflow.runName" = "lstm" and params.`batch size` = 64 and metrics.`categ acc` > 0.8

BiLSTM. BS=64, e=50, hs=[100,50]. categ acc = 0.844, 0.822 , 0.814
    0.826 +- 0.0127

LSTM attn. bs=64, e=50, hs=[100, 50] categ acc= 0.837, 0.82, 0.819
    0.8255 +- 0.008
    
BiLSTM Attn bs=64, e=50, hs=[100, 50] categ acc= 0.838, 0.837, 0.816
    0.830 +-0.010

LSTM bs=64, e=50, hs=[100, 50] categ acc=
    0.826 +-0.002
    
RNN bs=64, e=50, hs=[100, 50] categ acc=
    0.818 +-0.012
    
BiLSTM Attn bs=128, e=50, hs=[100, 50] categ acc= 0.836, 0.831, 0.823
    0.83 +- 0.005
    
LSTM attn. bs=128, e=50, hs=[100, 50] categ acc= 0.839, 0.831, 0.818
    0.829 +- 0.0086
        
     
#Changelog:
- 2020-08-14: Update dataset with enhanced rivetter labels
- 2020-08-12: Update dataset with twice the amount of labeled data; enhanced labels.
- 2020-07-29: Update data loader and notebooks

Known issues:
- 2020-08-12: Enforcing window lenghts when segmenting rivetter data lets two very short windows (label -1) slip through. These cause problems and need to be filtered out: 'filter_labels(..)'

License:
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
