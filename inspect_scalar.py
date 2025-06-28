import pickle

# Load the scaler
scaler = pickle.load(open("normalizer.pkl", "rb"))

# Print number of features expected
print("Scaler expects", scaler.n_features_in_, "features.")

# If available, print feature names (works if scaler was fitted on DataFrame)
try:
    print("\nFeature names:")
    print(list(scaler.feature_names_in_))
except AttributeError:
    print("\nFeature names are not stored in the scaler.")
