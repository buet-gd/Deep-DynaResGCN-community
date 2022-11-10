# Overlapping Community Detection using Dynamic Dilated Aggregation in Deep Residual GCNs



## Usage
<!-- The main algorithm and other utilities are implemented in the `nocd` package that can be installed as
```bash
python setup.py install
``` -->

Create a virtual environment in Miniconda3 as follows
```bash
conda env create -f environment.yaml
```
Then activate the environment as follows
```bash
conda activate deepnocd
```
For testing installation, run the following command
```bash
python run-deep.py
```

Jupyter notebooks [interactive-G.ipynb](interactive-deep-G.ipynb) and [interactive-X.ipynb](interactive-deep-X.ipynb) contain the code for training the model and analyzing the results interactively.




## Requirements
```
numpy=1.16.4
pytorch=1.2.0
scipy=1.3.1
networkx
matplotlib
scikit-learn
tqdm
```


<!-- ## Cite
Please cite our paper if you use the code or the datasets in your own work
```
@article{
    muttakin2022dynaResGCNOverlapping,
    title={Overlapping Community Detection with Graph Neural Networks},
    author={Oleksandr Shchur and Stephan G\"{u}nnemann},
    journal={Deep Learning on Graphs Workshop, KDD},
    year={2022},
}
``` -->

## Acknowledgments ##
Our codebase is built and extended upon the awesome [NOCD](https://github.com/shchur/overlapping-community-detection) repository.
