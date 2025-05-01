# To run the app (without re-training the models)

From the root directory of the project, run 

```
docker compose up --build
```

This will handle the Python installation and the necessary packages to build and run the application.

Visit http://127.0.0.1:8000.

# To run the entire process from scratch (downloading data, extracting features, training models, etc.)

Install the requirements (Tested with Python 3.9.6):

```
pip install -r requirements.txt
```

To download and organize the dataset, from the root directory:
```
python3 src/scripts/data.py
```

To extract features and spectograms, from the root directory

```
python3 src/scripts/feature_extraction.py
```

Then, run the notebooks in `src/scripts/notebooks`:

* `eda.ipynb` does some data exploration
* `feature_extraction.ipynb` performs and visualizes feature extraction on a sample file
* `train_models.ipynb` trains various classification models
* `cnn_vit.ipynb` trains the CNN and ViT models