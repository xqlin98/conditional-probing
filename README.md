# Data valuation using the conditional probing
Here we try to compute the value of data using conditional probing as the utility function. In the original paper, conditional probing is defined as:

$I_{\mathcal{V}}(\phi(X) \xrightarrow{} Y\mid B) = H_{\mathcal{V}}(Y\mid B) - H_{\mathcal{V}}(Y\mid B,\phi(x)) \ .$

And the way that the original paper interprets it is that the information that the pretrained representations have which are not at the original embedding $B$. And the function family specifies the function family that tells us how much the usable information we have about Y when extracting from representation using the function in the function family. And the original conditional probing paper uses $f=\argmin_{f\in\mathcal{V}} \frac{1}{|D_{train}|} \sum_{(x,y)\in D_{train}} \ell(f(x),y)$ to get $f$ and estimate the usable information. However, when different data points are used in the training dataset, the usable information can be different. Intuitively, some data points can make resultant model $f$ extract more usable information than some others. Therefore, we propose to use the following utility function to define the value of a coalition of data points $S$.

$U(S) = I_{\mathcal{V}_{S}}(\phi(X) \xrightarrow{} Y\mid B), \mathcal{V}_S = \{g: g=\argmin_{f\in\mathcal{V}} \frac{1}{|S|} \sum_{(x,y)\in S} \ell(f(x),y)\}$.

Where $\mathcal{V}_S$ is a set of minimizers of loss function on the data subset $S$. Empirically, we just let $\mathcal{V}$ be a single element set that contains a single model trained on the data subset $S$.

With this utility function $U(S)$, we can use data Shapley to compute the data value of each data point. 

# Configuration files
The example configuration file is put under `configs/dshap/layer0-0.yaml`


# Prepare the conda environment

```
conda create -n cprob -f environment.yml
```

# Dowload the data needed
Download the `data` folder in [CodaLab executable paper](https://worksheets.codalab.org/worksheets/0x46190ef741004a43a2676a3b46ea0c76) and put the whole folder under `./`

# How to run the code

```
python vinfo/experiment.py configs/dshap/layer0-0.yaml
```

The running result will be stored at `configs/dshap/layer0-0.yaml.results` folder.