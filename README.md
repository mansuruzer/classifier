# Classifier

## Installation and Usage

First, open a PowerShell window on your computer and copy and paste the following command:

```powershell
irm https://ollama.com/install.ps1 | iex
```

Wait until the installation is complete. Then, run the following command to download and run the AI model:

```powershell
ollama run qwen2.5:7b
```

You can use another model depending on your computer's hardware specifications. After the model installation is complete, open the Python file and replace the CSV filename with the name of the CSV file on your computer. For example:

```python
data = pd.read_csv("your_file.csv")
```

Replace `"your_file.csv"` with the actual name of your CSV file. Finally, run the Python file to start the classifier.

> **Important:** To test whether the model is working correctly, set `TEST_MODE = False` on line 33.
