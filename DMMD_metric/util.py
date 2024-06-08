from torch.utils.data import Dataset, DataLoader
import numpy as np
import tqdm
from dmmd_dataset import DMMDDataset 
import distance
import embedding

def compute_embeddings_for_dir(img_dir, embedding_model, batch_size):
    dataset = DMMDDataset(img_dir, reshape_to=embedding_model.input_image_size)
    count = len(dataset)
    print(f"Calculating embeddings for {count} images from {img_dir}.")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    embeddings = []
    
    for batch in tqdm.tqdm(dataloader, total=count // batch_size):
        image_batch = batch.numpy()
        image_batch = image_batch / 255.0
        cur_embs = np.asarray(embedding_model.get_embedding(image_batch))  
        embeddings.append(cur_embs)
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

def compute_dmmd(ref_dir, eval_dir, batch_size):
    embedding_model = embedding.DinoEmbedding()
    ref_embs = compute_embeddings_for_dir(ref_dir, embedding_model, batch_size).astype("float32")
    eval_embs = compute_embeddings_for_dir(eval_dir, embedding_model, batch_size).astype("float32")
    val = distance.mmd(ref_embs, eval_embs)
    return val.numpy()