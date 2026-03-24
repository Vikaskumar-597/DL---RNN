# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset

Develop a Recurrent Neural Network (RNN) model to predict stock prices based on historical data. Preprocess the dataset by handling missing values and normalizing time-series inputs. Train and evaluate the model to accurately forecast future stock trends.Develop a Recurrent Neural Network (RNN) model to predict stock prices based on historical data. Preprocess the dataset by handling missing values and normalizing time-series inputs. Train and evaluate the model to accurately forecast future stock trends.
<img width="618" height="655" alt="image" src="https://github.com/user-attachments/assets/afb54d5f-2047-468b-9522-62e0af23fe65" />

## DESIGN STEPS
### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 

Convert data to tensors and set up DataLoader.

### STEP 3: 

Define the RNN model architecture.

### STEP 4: 

Summarize, compile with loss and optimizer.

### STEP 5: 

Train the model with loss tracking.

### STEP 6: 

Predict on test data, plot actual vs. predicted prices.



## PROGRAM

### Name : VIKASKUMAR M

### Register Number : 212224220122

```python
class RNNModel(nn.Module):
    def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
      super(RNNModel,self).__init__()
      self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
      self.fc=nn.Linear(hidden_size,output_size)

    def forward(self,x):
      out,_=self.rnn(x)
      out=self.fc(out[:,-1,:])
      return out



# Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses=[]
    model.train()
    for epoch in range(epochs):
      total_loss=0
      for x_batch,y_batch in train_loader:
        x_batch,y_batch = x_batch.to(device),y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      train_losses.append(total_loss/len(train_loader))
      print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

    # Plot training loss
    print('Name: Vikaskumar M')
    print('Register Number: 212224220122')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()


```

## OUTPUT

## Training Loss Over Epochs Plot

<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/b2d7de02-db9d-45ca-befb-12c58148dc36" />


## True Stock Price, Predicted Stock Price vs time

<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/f18a7aff-f0fc-4521-b661-39bb0791d841" />


### Predictions
<img width="342" height="66" alt="image" src="https://github.com/user-attachments/assets/e5058181-07f4-42af-a9db-78dae49683ed" />


## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
