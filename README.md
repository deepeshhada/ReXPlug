# ReXPlug (ACM SIGIR, 2021)
## Explainable Recommendation using Plug and Play Language Model

#### Preprocessing
Preprocess the raw dataset in a form ReXPlug ingests. The raw dataset must be a JSON file with the same name as dataset_name and must be zipped in dataset_path. For example, AmazonDigitalMusic.json is zipped in the raw_datasets directory as AmazonDigitalMusic.zip. If you're using a non-Amazon dataset, please make sure that the JSON file has only four fields, namely, `'userId', 'itemId', 'review', 'rating'`, in that order. The `truncate_after` flag determines the number of interactions to be used for training the Discriminator.

`python preprocess.py --dataset_name="AmazonDigitalMusic" --dataset_path="./data/raw_datasets/AmazonDigitalMusic.zip" --seed=1234 --truncate_after=100000`

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

`python test.py --length=50 --num_iterations=1 --temperature=1 --sample --gamma=1 --gm_scale=0.875 --kl_scale=0.01 --num_reviews=5`

------------

