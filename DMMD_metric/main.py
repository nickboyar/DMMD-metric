from absl import app
import distance
import embedding
import util
import numpy as np

def main(argv):
    _, dir1, dir2, batch_size = argv    
    print("The DMMD value is: "f" {util.compute_dmmd(dir1, dir2, int(batch_size)):.3f}")

if __name__ == "__main__":
    app.run(main)