import torch
import torch.nn.functional as F
import copy


def train_step(model, batch, loss_fn, device=None):
    X, y = batch
    if device is not None:
        X, y = X.to(device), y.to(device)
    y_hat = model(X)
    return loss_fn(y_hat, y)

def deep_to(b, device):
    """Moves batch of an arbitrary structure to the specified device.
    
    Assumes that the batch is a collection (tuple, list or dict)
    of tensors (which is typically the case).
    """

    def move(b):
        if torch.is_tensor(b):
            return b.to(device)
        for tp in (list, tuple):
            if isinstance(b, tp):
                return tp(move(x) for x in b)
        if isinstance(b, dict):
            return {k: move(v) for k, v in b.items()}
            
        raise ValueError('Unsupported batch type')

    return move(b)

def train(model,
          loss_fn, 
          *,
          dataloader=None,
          data=None,
          val_dataloader=None,
          val_data=None,
          batch_size=None,
          optimizer='adam',
          lr=1e-3,
          weight_decay=0,
          max_iter=100,
          train_step=train_step,
          early_stopping_patience=10,
          verbose=True,
          log_every=20,
          device=None, 
          save_best_on_train_loss=False):

    if isinstance(device, str):
        try:
            device = torch.device(device)
        except Exception:
            raise ValueError(f'Invalid device string: "{device}". Expected "cuda" or "cpu".')
    elif device is not None and not isinstance(device, torch.device):
        raise TypeError(f'device must be str ("cuda"/"cpu"), torch.device, or None, got {type(device)}')

    if device is not None:
        model = model.to(device)

    if dataloader is not None and data is not None or \
        dataloader is None and data is None:
        raise ValueError('Exactly one of "data" and "dataloader" must be specified')

    if val_dataloader is not None and val_data is not None:
        raise ValueError('Only one of "val_data" and "val_dataloader" can be specified')
    
    if dataloader is None:
        if batch_size is None:
            batch_size = len(data)
        dataloader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
    
    if val_dataloader is None and val_data is not None:
        if batch_size is None:
            batch_size = len(val_data)
        val_dataloader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)
    
    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    else:
        pass

    best_val_loss = float('inf')
    best_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(max_iter):
        model.train()
        epoch_loss = 0.0

        for j, batch in enumerate(dataloader):
            optimizer.zero_grad()
            if device is not None:
                batch = deep_to(batch, device)
            loss = train_step(model, batch, loss_fn, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

        epoch_loss /= (j + 1)

        val_loss = None
        if val_dataloader is not None:
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for j, batch in enumerate(val_dataloader):
                    if device is not None:
                        batch = deep_to(batch, device)
                    val_loss_total += train_step(model, batch, loss_fn, device).item()
                val_loss = val_loss_total / (j + 1)

            if verbose and (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch+1}: train_loss = {epoch_loss:.4f}, val_loss = {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            if verbose and (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch+1}: train_loss = {epoch_loss:.4f}")
                
            if save_best_on_train_loss and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_state = copy.deepcopy(model.state_dict())

    if val_dataloader is not None or save_best_on_train_loss:
        model.load_state_dict(best_model_state)
        
