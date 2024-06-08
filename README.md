# DMMD-metric
Implementation of the DMMD metric obtained as a result of research conducted on the basis of the [framework](https://github.com/nickboyar/create-and-test-generative-model-metric).
You can use the metric as an alternative to FID to evaluate the quality of generated images.

The metric is implemented based on the PyTorch framework.

## Setup

Install dependencies 

```bash
cd DMMD_metric
pip install -r requirements.txt
```

## Usage

```bash
python main.py /path/to/reference/images /path/to/eval/images batch_size
```
