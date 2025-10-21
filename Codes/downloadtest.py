import kagglehub

# Download latest version
path = kagglehub.dataset_download("/home/karan-sankla/Documents")

print("Path to dataset files:", path)