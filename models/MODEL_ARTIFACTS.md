# Model Artifacts

Model files are generated locally and ignored by Git. They are not committed because checkpoints and serialized model files can be large and are environment-specific.

Expected local model files:

```text
models/classical/classical_lbp_hsv_hog_svm.joblib
models/cnn/best_cnn_model_v2.pth
```

If this repository is cloned on a new machine, these files must be trained locally or copied manually from a previous local run.

The image demo, video demo, and Streamlit app require these model files to run new predictions. Existing generated outputs may be viewable if they were supplied separately, but regeneration needs the local model artifacts.

Do not upload large model files unless the professor's submission rules explicitly allow or require them.
