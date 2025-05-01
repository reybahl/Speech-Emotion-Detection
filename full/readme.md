# To run the app (without re-training the models)

From the root directory of the project, run 

```
docker compose up --build
```

This will handle the Python installation and the necessary packages to build and run the application.

Once the docker container has been built, visit http://127.0.0.1:8000.

# To run the entire process from scratch (downloading data, extracting features, training models, etc.)

Install the requirements (Tested with Python 3.9.6):

```
pip install -r requirements.txt
```

The data used has been provided in this repository; therefore, it is not necessary to run the following command. However, if you wish to download and organize the dataset from scratch, run this command from the root directory:
```
python3 src/scripts/data.py
```


*This step is also optional as the features have been provided in the repository --*
To extract features and spectograms, run

```
python3 src/scripts/feature_extraction.py
```

Then, run the notebooks in `src/scripts/notebooks`:

* `eda.ipynb` does some data exploration / visualization
* `feature_extraction.ipynb` performs and visualizes feature extraction on a sample file
* `train_models.ipynb` trains various classification models and saves them
* `cnn_vit.ipynb` trains the CNN and ViT models
* `testing.ipynb` tests all of the trained models using the saved files

*Note: depending on where you are running the files, you may need to change the directories of files accessed in these notebooks.* If you are running locally where the root directory is organized as below:

```
data/
models/
src/
    notebooks/
        train_models.ipynb
        ...
```

then the file directories provided in the notebooks should be correct as they currently are.