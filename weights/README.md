# Model weights

Run the model setup command from the root README to populate this folder.
The runtime loads the checkpoint from clip/ and can run offline after setup.
Include the weights in the final handoff if the official rules permit this model.

`action_adapter.npz` contains 63 visual prototypes fitted only on four EPIC training videos. The vocabulary signature is checked before use. See docs/recognition/README.md for provenance, fitting commands, evaluation and noncommercial data-use attribution.
