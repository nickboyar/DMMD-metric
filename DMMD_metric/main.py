"""The main entry point for the CMMD calculation."""

from absl import app
from absl import flags
import distance
import embedding
import io_util
import numpy as np
import pandas as pd
import timm

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "Batch size for embedding generation.")
_MAX_COUNT = flags.DEFINE_integer("max_count", -1, "Maximum number of images to read from each directory.")
_REF_EMBED_FILE = flags.DEFINE_string(
    "ref_embed_file", None, "Path to the pre-computed embedding file for the reference images."
)


import numpy as np
import torch
import os
import json

from torchvision.transforms import functional as F


def compute_dmmd(ref_dir, eval_dir, ref_embed_file=None, batch_size=32, max_count=-1):

    if ref_dir and ref_embed_file:
        raise ValueError("`ref_dir` and `ref_embed_file` both cannot be set at the same time.")
    embedding_model = embedding.DinoEmbeddingModel()
    if ref_embed_file is not None:
        ref_embs = np.load(ref_embed_file).astype("float32")
    else:
        ref_embs = io_util.compute_embeddings_for_dir(ref_dir, embedding_model, batch_size, max_count).astype(
            "float32"
        )
    eval_embs = io_util.compute_embeddings_for_dir(eval_dir, embedding_model, batch_size, max_count).astype("float32")
    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()
    
    

def main(argv):
    if len(argv) != 3:
        raise app.UsageError("Too few/too many command-line arguments.")
    _, dir1, dir2 = argv
    
    print(
        "The CMMD value is: "
        f" {compute_dmmd(dir1, dir2, _REF_EMBED_FILE.value, _BATCH_SIZE.value, _MAX_COUNT.value):.3f}"
    )
    

if __name__ == "__main__":
    app.run(main)
