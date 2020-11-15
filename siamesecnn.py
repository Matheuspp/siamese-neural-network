class SiameseNetwork(nn.Module):
     def __init__(self):
          super(SiameseNetwork, self).__init__()
          # Setting up the Sequential of CNN Layers
          self.cnn1 = nn.Sequential( 
          nn.Conv2d(1, 96, kernel_size=11,stride=1),
          nn.ReLU(inplace=True),
          nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
          nn.MaxPool2d(3, stride=2),

          nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
          nn.ReLU(inplace=True),
          nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
          nn.MaxPool2d(3, stride=2),
          nn.Dropout2d(p=0.3),

          nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(3, stride=2),
          nn.Dropout2d(p=0.3),
          )

          # Defining the fully connected layers
          self.fc1 = nn.Sequential(
          # First Dense Layer
          nn.Linear(30976, 1024),
          nn.ReLU(inplace=True),
          nn.Dropout2d(p=0.5),
          # Second Dense Layer
          nn.Linear(1024, 128),
          nn.ReLU(inplace=True),
          # Final Dense Layer
          nn.Linear(128,2))

     def forward_once(self, x):
          # Forward pass 
          output = self.cnn1(x)
          output = output.view(output.size()[0], -1)
          output = self.fc1(output)
          return output

     def forward(self, input1, input2):
          # forward pass of input 1
          output1 = self.forward_once(input1)
          # forward pass of input 2
          output2 = self.forward_once(input2)
          # returning the feature vectors of two inputs
          return output1, output2
         
class ContrastiveLoss(torch.nn.Module):

      def __init__(self, margin=2.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

      def forward(self, output1, output2, label):
            # Find the pairwise distance or eucledian distance of two output feature vectors
            euclidean_distance = F.pairwise_distance(output1, output2)
            # perform contrastive loss calculation with the distance
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            return loss_contrastive
            
      def oneshot(model,img1,img2):
       # Gives you the feature vector of both inputs
       output1,output2 = model(img1.cuda(),img2.cuda())
       # Compute the distance 
       euclidean_distance = F.pairwise_distance(output1, output2)
       #with certain threshold of distance say its similar or not
       if eucledian_distance > 0.5:
               print("Orginal Signature")
       else:
               print("Forged Signature")
