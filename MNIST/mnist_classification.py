import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np

def plot_sample_images(image_data):
    # Convert to numpy array and ensure proper shape
    image = np.array(image_data).reshape(28, 28)
    plt.imshow(image, cmap='gray')  
    plt.axis('off')
    plt.show()

# Fetch MNIST data
print("Fetching MNIST dataset (this may take a moment)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=True)
X, y = mnist.data, mnist.target

# Display first 20 pixel values of the first image
print("\nFirst 20 pixel values of the first image:")
print(X.iloc[0][:20])

# Display summary statistics
print("\nImage statistics:")
print("Min pixel value:", X.iloc[0].min(), "Max pixel value:", X.iloc[0].max())

# Plot the first image
print("\nDisplaying the first digit image...")
some_image = X.iloc[0].values  # Get the first image as numpy array
plot_sample_images(some_image)
