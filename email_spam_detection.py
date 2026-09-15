from pathlib import Path
import urllib.request, zipfile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"; DATA.mkdir(exist_ok=True)
FILE=DATA/"SMSSpamCollection"
URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

def load_data():
    if not FILE.exists():
        archive=DATA/"smsspamcollection.zip"
        print("Downloading SMS Spam Collection from UCI...")
        urllib.request.urlretrieve(URL,archive)
        with zipfile.ZipFile(archive) as z: z.extractall(DATA)
        archive.unlink(missing_ok=True)
    df=pd.read_csv(FILE,sep="\t",header=None,names=["label","message"],encoding="utf-8")
    return df.dropna(subset=["label","message"]).drop_duplicates("message").query("label in ['ham','spam']")

def main():
    df=load_data()
    X=df["message"]; y=df["label"].map({"ham":0,"spam":1})
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    models={
        "Multinomial Naive Bayes":MultinomialNB(),
        "Logistic Regression":LogisticRegression(max_iter=1000,random_state=42),
        "SVM (linear)":SVC(kernel="linear"),
        "KNN":KNeighborsClassifier(),
        "Decision Tree":DecisionTreeClassifier(random_state=42)}
    rows=[]; pipes={}
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    for name,model in models.items():
        pipe=Pipeline([("vectorizer",CountVectorizer(stop_words="english")),("model",model)])
        pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        rows.append({"Model":name,"Test Accuracy":accuracy_score(yte,pred),
                     "Macro Precision":precision_score(yte,pred,average="macro",zero_division=0),
                     "Macro Recall":recall_score(yte,pred,average="macro",zero_division=0),
                     "Macro F1":f1_score(yte,pred,average="macro",zero_division=0),
                     "5-Fold CV Accuracy":cross_val_score(pipe,Xtr,ytr,cv=cv,scoring="accuracy").mean()})
        pipes[name]=pipe
    result=pd.DataFrame(rows).sort_values("Test Accuracy",ascending=False)
    print(result.to_string(index=False))
    best=result.iloc[0]["Model"]; pipe=pipes[best]; pred=pipe.predict(Xte)
    print("\nSelected:",best); print(classification_report(yte,pred,target_names=["ham","spam"]))
    ConfusionMatrixDisplay(confusion_matrix(yte,pred),display_labels=["ham","spam"]).plot()
    plt.title(f"Confusion Matrix - {best}"); plt.tight_layout()
    plt.savefig(ROOT/"confusion_matrix.png",dpi=300); plt.show()
    for msg in ["Congratulations! You've WON a $1000 Walmart gift card. Click here to claim now!!!",
                "Hey, are we still on for lunch tomorrow at 1pm?",
                "URGENT: Your account has been suspended. Verify your details immediately.",
                "Can you send me the notes from today's class when you get a chance?"]:
        p=pipe.predict([msg])[0]
        print(f"\n{msg}\nPrediction: {'SPAM' if p else 'HAM'}")
if __name__=="__main__": main()
