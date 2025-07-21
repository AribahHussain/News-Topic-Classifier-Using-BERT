# 📰 News Topic Classifier Using BERT

## 📌 Overview

This project uses a fine-tuned BERT model to automatically classify news text into one of four categories: **World**, **Sports**, **Business**, and **Sci/Tech**. It leverages transformer-based NLP to understand the context of the input and predict the most relevant topic with high accuracy.

---

## 🔤 Description

This project uses a fine-tuned BERT model to classify news articles into categories like World, Sports, Business, and Sci/Tech. It processes text input and predicts the topic with high accuracy, enabling fast and consistent news classification for real-world applications.

---

## 🎯 Objective

The goal of this project is to:
- Automate topic tagging of news content
- Improve the accuracy and consistency of classification
- Enable real-time news categorization through an interactive interface

---

## ⚙️ Features

- ✅ BERT-based fine-tuned text classifier
- ✅ Multi-class prediction (4 categories)
- ✅ Tokenization and inference using Hugging Face Transformers
- ✅ Confidence scores via softmax
- ✅ Real-time prediction with a Gradio interface
- ✅ Simple and reusable model pipeline

---

## 🧠 How It Works

1. The user inputs a news headline or short article.
2. The text is tokenized using a BERT tokenizer.
3. The model performs inference and outputs class logits.
4. Softmax converts logits to class probabilities.
5. The class with the highest probability is selected as the final prediction.
6. Gradio displays the prediction and confidence scores.

---

## 🖥️ Gradio Interface

Gradio is used to build an interactive UI that allows users to:
- Enter text into a simple textbox
- See predicted news topic in real time
- View confidence scores of all classes
- Run the app without any code (just run the cell or script)

> Gradio makes the model demo-ready, easy to test, and ideal for presentation or deployment.

---

## 📊 Dataset

**Dataset Used**: [AG News Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

- 120,000 training samples and 7,600 test samples
- 4 categories: World, Sports, Business, Sci/Tech
- Each sample includes a title and a short description

---

## 🧰 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Gradio
- Pandas, NumPy

---

## 📈 Output Example

**Input**:  
> "Microsoft shares soar after strong quarterly earnings"

**Prediction**:  
> **Business** (Confidence: 91%)

**Other Classes**:  
- Sci/Tech: 5%  
- World: 3%  
- Sports: 1%

---

## 💡 Applications

- News apps for topic-based filtering
- Content recommendation engines
- Editorial categorization
- Social media/news monitoring tools

---

## 🚀 Future Improvements

- Expand categories using multi-label classification
- Apply to multilingual news content
- Deploy as a live web service or API
- Extend to long-form articles and summaries

---

## 📎 License

This project is licensed under the MIT License.  
Feel free to use, modify, and share with proper credit.

---

## 🙌 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [AG News Dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
- [Gradio](https://www.gradio.app/)

---

