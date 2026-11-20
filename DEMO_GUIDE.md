# CottonLeaf AI demo guide

Allow about five minutes. The published benchmark and architecture walkthrough
work with the core dependencies alone. Dataset scanning requires local images.
Training and classification require the optional TensorFlow dependencies, and
classification requires a compatible checkpoint.

## Start the app

On Windows, run `.\run_app.bat` from the repository folder. On other platforms,
activate your Python 3.11 virtual environment and run:

```bash
python -m streamlit run app.py --server.headless true
```

Open the URL printed in the terminal. See [Windows and PyCharm setup](README.md)
for installation and interpreter details.

## Walk through the dashboard

1. **Overview.** Introduce the four class labels and the two feature branches.
   The 99.41% card is the paper's final-classification benchmark.
2. **Model Comparisons.** Show the published feature-extraction and final
   classification charts separately. Open **Our Experiment** to show local
   measurements or the pending state. Use Plotly's camera icon to export a
   PNG or the table's download button to export CSV.
3. **Model Architecture.** Trace the hybrid model from preprocessing to
   softmax. Open the attention and stem tabs to explain channel scaling and
   spatial feature extraction.
4. **Dataset & Training.** If local images are available, scan the class
   folders and show the usable splits. Use an already completed run to show
   real history in the Fast Demo or Full Hybrid card. A short Fast Demo can
   demonstrate training with a compatible dataset. For a presentation without
   training, click **Preview 50-Epoch Research Target**. The labeled simulation
   takes about 40 seconds and finishes at the published 99.41% target.
5. **Upload & Classification.** Upload a JPEG or PNG. With a compatible
   checkpoint, click **Classify image** and show the model name and class
   probabilities. Without one, the image preview still works and the
   classification button stays disabled.
6. **Evaluation Metrics.** Show the published five-fold references. Select
   a completed local run to show actual confusion counts, ROC curves, and
   complexity measurements. The local section displays **Pending evaluation**
   when measurements are unavailable.

Keep the paper's results attributed to Sailaja Madhu and V. Ravisankar.
Only completed local runs with saved artifacts provide locally measured results.

[Captured overview at desktop width](docs/CottonLeaf-Overview.png)
