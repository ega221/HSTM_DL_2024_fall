import argparse

# Без этого у меня почему-то не работает
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class MultiLayerPerceptron(nn.Module):

    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(360, 180)
        self.dropout1 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(180)
        self.fc2 = nn.Linear(180, 90)
        self.dropout2 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(90)
        self.fc3 = nn.Linear(90, 3)

    def forward(self, x: torch.Tensor):
        out1 = torch.relu(self.bn1(self.fc1(x)))
        outDropped1 = self.dropout1(out1)
        out2 = torch.relu(self.bn2(self.fc2(outDropped1)))
        outDropped2 = self.dropout2(out2)
        out3 = self.fc3(outDropped2)
        return out3

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            probabilities = torch.softmax(self.forward(x), dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        return predicted_classes

def load_data(train_csv: str, val_csv: str, test_csv: str):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    y_train = train_df["order0"]
    y_val = val_df["order0"]

    X_train = train_df.drop(["order0", "order1", "order2"], axis=1)
    X_val = val_df.drop(["order0", "order1", "order2"], axis=1)

    X_test = test_df

    return X_train, y_train, X_val, y_val, X_test

def preprocess_data(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame):
    y_train = torch.tensor(y_train.values, dtype=torch.int64)
    X_train = torch.tensor(X_train.values, dtype=torch.float32)

    y_val = torch.tensor(y_val.values, dtype=torch.int64)
    X_val = torch.tensor(X_val.values, dtype=torch.float32)

    X_test = torch.tensor(X_test.values, dtype=torch.float32)

    return X_train, y_train, X_val, y_val, X_test

def evaluate(model: MultiLayerPerceptron, X: torch.Tensor, y: torch.Tensor):
    model.eval()
    predictions = model.predict(X)
    accuracy = accuracy_score(y_true=y.numpy(), y_pred=predictions.numpy())
    confusion_mtrx = confusion_matrix(y_true=y.numpy(), y_pred=predictions.numpy())
    return predictions, accuracy, confusion_mtrx

def init_model(learning_rate: float):
    model = MultiLayerPerceptron()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    return model, criterion, optimizer

def train(model: MultiLayerPerceptron, criterion: nn.Module, optimizer: optim.Optimizer, X_train: torch.Tensor, y_train: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, epochs: int, batch_size: int):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            X_batch = X_train[i: i + batch_size]
            y_batch = y_train[i: i + batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss /= X_train.size(0) // batch_size
        predictions_val, accuracy_val, confusion_matrix = evaluate(model, X_val, y_val)
        print(f"-----------------------------------------------------------")
        print(f"Epoch {epoch+1}/{epochs}, \nLoss: {train_loss}, \nValidation Accuracy: {accuracy_val}, \nConfusion Matrix: \n{confusion_matrix}")
        print(f"***********************************************************")
    return model


def main(args: argparse.Namespace):
    X_train, y_train, X_val, y_val, X_test = load_data(args.train_csv, args.val_csv, args.test_csv)
    X_train, y_train, X_val, y_val, X_test = preprocess_data(X_train, y_train, X_val, y_val, X_test)
    model, criterion, optimizer = init_model(args.lr)
    train(model, criterion, optimizer, X_train, y_train, X_val, y_val, args.num_epoches, args.batch_size)
    predictions_test = model.predict(X_test).numpy()
    pd.Series(predictions_test).to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_csv', default='./data/train.csv')
    parser.add_argument('--val_csv', default='./data/val.csv')
    parser.add_argument('--test_csv', default='./data/test.csv')
    parser.add_argument('--out_csv', default='./data/submission.csv')
    parser.add_argument('--lr', default=4.10696400890577424e-05)
    parser.add_argument('--batch_size', default=1024)
    parser.add_argument('--num_epoches', default=100)

    args = parser.parse_args()
    main(args)
