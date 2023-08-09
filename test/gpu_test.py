import torch   
print(torch.__version__)
  
torch.cuda.is_available()   
 
torch.cuda.device(0)
torch.cuda.device_count()   

torch.cuda.get_device_name(0)