# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

 Load and Preprocess Data

### STEP 2: 


Feature Scaling and Data Split

### STEP 3: 

Convert Data to PyTorch Tensors

### STEP 4: 

Define the Neural Network Model

### STEP 5: 

 Train the Model

### STEP 6: 


Evaluate and Predict


## PROGRAM

### Name:R.SUBHASHRI

### Register Number:212223230219

```python
# Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,8)
        self.fc4=nn.Linear(8,4)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x

     
        
# Initialize the Model, Loss Function, and Optimizer

model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)
# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()
  for epoch in range(epochs):
    for inputs,labels in train_loader:
      optimizer.zero_grad()
      outputs=model(inputs)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information

<img width="1147" height="629" alt="image" src="https://github.com/user-attachments/assets/6063511f-5f63-430c-a9a5-b2f0275bf414" />


### OUTPUT

## Confusion Matrix

<img width="585" height="472" alt="image" src="https://github.com/user-attachments/assets/43a9f024-ffb1-4938-8b27-0fb9b693ce12" />


## Classification Report
<img width="642" height="345" alt="image" src="https://github.com/user-attachments/assets/a4efb025-fbfe-432b-a531-aaff6284bd5e" />


### New Sample Data Prediction
<img width="350" height="85" alt="image" src="https://github.com/user-attachments/assets/f373d172-eb14-40b8-8d05-e275fa1f98f1" />


## RESULT
This program has been executed successfully.
