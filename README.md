# ReXPlug
## Explainable Recommendation using Plug and Play Language Model

A [Google Colab](https://colab.research.google.com/drive/1mSkbVKV7Jqu0UfrkgDKVpXKxA8gsQaxH?usp=sharing) notebook is available for demonstration.

------------

#### Preprocessing
Preprocessing currently downloads the already preprocessed splits and files needed for training ReXPlug. Following is an example:

`python preprocess.py --dataset_name="AmazonDigitalMusic" --split_idx="1" --truncate_after=100000`

------------
#### Training ReXPlug
**1. Training RRCA.**
For smaller datasets like Digital Music and Clothing, the training takes around 50-100 epochs. For larger datasets like BeerAdvocate and Yelp, training takes ~30 epochs. The code maintains a patience of 15 epochs.

`python train_rrca.py --dataset_name="AmazonDigitalMusic" --batch_size_rrca=256 --learning_rate_rrca=0.002 --num_epochs_rrca=150`

**2. Training Discriminator.**
The following command trains the discriminator. 2-3 epochs usually suffice.

`python train_discrim.py --dataset_name="AmazonDigitalMusic" --pretrained_model="gpt2-medium" --batch_size=64 --learning_rate=0.002 --epochs=3`

------------
#### Generating Reviews
The following command generates reviews from the trained ReXPlug.

`python test.py --length=150 --num_iterations=1 --temperature=1 --sample --gamma=1 --gm_scale=0.875 --kl_scale=0.01`

------------

