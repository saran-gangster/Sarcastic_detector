# Sarcastic_detector
A Simple neural network that detects sarcasm

## Installation
```
pip3 install tensorflow
pip3 install numpy
pip3 install wget
git clone https://github.com/saran-gangster/Sarcastic_detector.git
cd Sarcastic_detector
```

### Train the model

```
python train.py
```

### Test the model

```
python Classifier.py
```

### Sample results

```
You: I am busy right now, can I ignore you some other time?
This Sentence is Sarcastic
Probabilty:0.9322293996810913
You: Lead me not into temptation. I know the way.
This Sentence is Sarcastic
Probabilty:0.9322293996810913
You: Zombies eat brains. You're safe.
This Sentence is Sarcastic
Probabilty:0.9322293996810913
You: Find your patience before I lose mine.
This Sentence is Sarcastic
Probabilty:0.9322293996810913
You: If you find me offensive. Then I suggest you quit finding me.
This Sentence is Sarcastic
Probabilty:0.8312952518463135
You: You only live once, but if you do it right, once is enough.
This Sentence is not Sarcastic
Probabilty:0.9223993420600891
You: Without music, life would be a mistake.
This Sentence is not Sarcastic
Probabilty:0.9322293996810913
```

