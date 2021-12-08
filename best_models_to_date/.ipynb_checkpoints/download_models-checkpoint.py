import os
import wget


def download_model(model_fn, models_dir="models/"):
    base_url = "https://beautifeye-pub.s3.eu-west-1.amazonaws.com/Doc2Answer/models/"
    
    model_lpath = os.path.join(models_dir, model_fn)
    if os.path.exists(model_lpath):
        print("Model {} exists already".format(model_lpath))
    else:
        os.makedirs(models_dir, exist_ok=True)
        wget.download(url=base_url + model_fn, out=model_lpath)
        print("Model {} downloaded".format(model_lpath))   