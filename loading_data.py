import pandas as pd 

import kagglehub

# Download latest version
path = kagglehub.dataset_download("anshtanwar/global-data-on-sustainable-energy")

print("Path to dataset files:", path)