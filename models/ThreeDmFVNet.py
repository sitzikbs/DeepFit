import torch
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture

class ThreeDmFVNet(torch.nn.Module):
  def __init__(self, n_gaussians, num_classes, normalize= True):
    """
         3DmFV-Net architecture
         n_gaussians: number of gaussians in each axis
         num_classes: numer of classes for final fully connected layer
         normalize: flag for 3DmFV representation
    """
    super(ThreeDmFVNet, self).__init__()
    self.n_gaussians = n_gaussians

    gmm = get_3d_grid_gmm(subdivisions=[n_gaussians, n_gaussians, n_gaussians],
                                variance=np.sqrt(1.0 / n_gaussians))
    self.gmm = gmm
    self.normalize = normalize
    self.globalfeature = Global3DmFVFeature(n_gaussians)

    self.fc1 = torch.nn.Linear(2*2*2*512*3, 1024)
    self.bn1 = torch.nn.BatchNorm1d(1024)
    self.do1 = torch.nn.Dropout(0.7)
    self.fc2 = torch.nn.Linear(1024, 256)
    self.bn2 = torch.nn.BatchNorm1d(256)
    self.do2 = torch.nn.Dropout(0.7)
    self.fc3 = torch.nn.Linear(256, 128)
    self.bn3 = torch.nn.BatchNorm1d(128)
    self.do3 = torch.nn.Dropout(0.7)
    self.fc4 = torch.nn.Linear(128, num_classes)


  def forward(self, points):

      batch_size = points.shape[0]

      with torch.no_grad():
          ThreeDmFV = get_3DmFV_pytorch(points, self.gmm.weights_, self.gmm.means_,
                                              np.sqrt(self.gmm.covariances_), normalize=self.normalize)
      x = self.globalfeature(ThreeDmFV.view(-1, 20, self.n_gaussians, self.n_gaussians,  self.n_gaussians))

      x = F.relu(self.bn1(self.fc1(x)))
      x = self.do1(x)
      x = F.relu(self.bn2(self.fc2(x)))
      x = self.do2(x)
      x = F.relu(self.bn3(self.fc3(x)))
      x = self.do3(x)
      x = self.fc4(x)

      return x

class Global3DmFVFeature(torch.nn.Module):
    def __init__(self, n_gaussians):
        """
        architecture for learning the global feature from 3DmFV representation. It changes depending on the numbre of
        Gaussians
        :param n_gaussians: number of gaussians in 3DmFV representation
        """
        self.n_gaussians = n_gaussians
        if n_gaussians == 3:
            self.inception1 = Inception3D(2, 3, 20, 64)
            self.inception2 = Inception3D(2, 3, 64 * 3, 128)
            self.inception3 = Inception3D(2, 3, 128 * 3, 256)
            self.maxpool1 = None
            self.inception4 = Inception3D(2, 3, 256 * 3, 256)
            self.inception5 = Inception3D(2, 3, 256 * 3, 512)
            self.maxpool2 = None
        elif n_gaussians == 5:
            self.inception1 = Inception3D(3, 5, 20, 64)
            self.inception2 = Inception3D(3, 5, 64 * 3, 128)
            self.inception3 = Inception3D(3, 5, 128 * 3, 256)
            self.maxpool1 = torch.nn.MaxPool3d([3, 3, 3], 2)
            self.inception4 = Inception3D(2, 3, 256 * 3, 256)
            self.inception5 = Inception3D(2, 3, 256 * 3, 512)
            self.maxpool2 = None
        elif n_gaussians == 8:
            self.inception1 = Inception3D(3, 5, 20, 64)
            self.inception2 = Inception3D(3, 5, 64 * 3, 128)
            self.inception3 = Inception3D(3, 5, 128 * 3, 256)
            self.maxpool1 = torch.nn.MaxPool3d([2, 2, 2], 2)
            self.inception4 = Inception3D(3, 4, 256 * 3, 256)
            self.inception5 = Inception3D(3, 4, 256 * 3, 512)
            self.maxpool2 = torch.nn.MaxPool3d([2, 2, 2], 2)
        elif n_gaussians == 16:
            self.inception1 = Inception3D(4, 8, 20, 64)
            self.inception2 = Inception3D(4, 8, 64 * 3, 128)
            self.inception3 = Inception3D(4, 8, 128 * 3, 256)
            self.maxpool1 = torch.nn.MaxPool3d([2, 2, 2], 2)
            self.inception4 = Inception3D(3, 5, 256 * 3, 256)
            self.inception5 = Inception3D(3, 5, 256 * 3, 512)
            self.maxpool2 = torch.nn.MaxPool3d([2, 2, 2], 2)
            self.inception6 = Inception3D(2, 3, 512 * 3, 512)
            self.inception7 = Inception3D(2, 3, 512 * 3, 512)
            self.maxpool3 = torch.nn.MaxPool3d([2, 2, 2], 2)
        else:
            raise ValueError('Unsupported number of Gaussians')

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.maxpool1(x) if self.maxpool1 is not None else x
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.maxpool2(x) if self.maxpool2 is not None else x
        if self.n_guassians > 8:
            x = self.inception6(x)
            x = self.inception7(x)
            x = self.maxpool3(x)
        x = x.view(batch_size, -1)
        return x


class Inception3D(torch.nn.Module):
  def __init__(self, c1, c2, D_in, D_out):
    """
         3d Inception module for 3dmfv-net
         output feature is D_out*3
    """
    super(Inception3D, self).__init__()
    D_out_2 = int(D_out/2)
    # compute padding o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
    # from https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338 for input 8x8x8 and c1=3, c2=5

    self.conv1 = torch.nn.Conv3d(in_channels=D_in, out_channels=D_out, kernel_size=1, stride=1)
    self.bn1 = torch.nn.BatchNorm3d(D_out)
    self.conv2 = torch.nn.Conv3d(in_channels=D_in, out_channels=D_out_2, kernel_size=c1, stride=1, padding=1)
    self.bn2 = torch.nn.BatchNorm3d(D_out_2)
    self.conv3 =  torch.nn.Conv3d(in_channels=D_in, out_channels=D_out_2, kernel_size=c2, stride=1, padding=2)
    self.bn3 = torch.nn.BatchNorm3d(D_out_2)
    self.avgpool = torch.nn.AvgPool3d(kernel_size=c1, stride=1, padding=1)
    self.conv4 = torch.nn.Conv3d(in_channels=D_in, out_channels=D_out, kernel_size=1, stride=1)
    self.bn4 = torch.nn.BatchNorm3d(D_out)

  def forward(self, x):
    one_by_one = F.relu(self.bn1(self.conv1(x)))
    c1_by_c1 = F.relu(self.bn2(self.conv2(x)))
    c2_by_c2 = F.relu(self.bn3(self.conv3(x)))
    average_pooling = self.avgpool(x)
    average_pooling = F.relu(self.bn4(self.conv4(average_pooling)))
    return torch.cat((one_by_one, c1_by_c1, c2_by_c2, average_pooling), dim=1)


def get_3DmFV_pytorch(points, w, mu, sigma, normalize=True):
    """
       Compute the 3D modified fisher vectors given the gmm model parameters (w,mu,sigma) and a set of points
       :param points: B X N x 3 tensor of XYZ points
       :param w: B X n_gaussians tensor of gaussian weights
       :param mu: B X n_gaussians X 3 tensor of gaussian cetnters
       :param sigma: B X n_gaussians X 3 tensor of stddev of diagonal covariance
       :return: fv: B X 20*n_gaussians tensor of the fisher vector
       """
    n_batches = points.shape[0]
    n_points = points.shape[1]
    n_gaussians = mu.shape[0]
    D = mu.shape[1]
    if not torch.is_tensor(points):
        points = torch.from_numpy(points)
    device = points.device
    # Expand dimension for batch compatibility
    batch_sig = torch.from_numpy(sigma).unsqueeze(0).unsqueeze(0).repeat(n_batches, n_points, 1, 1).to(device)
    batch_mu = torch.from_numpy(mu).unsqueeze(0).unsqueeze(0).repeat(n_batches, n_points, 1, 1).to(device)
    batch_w = torch.from_numpy(w).unsqueeze(0).unsqueeze(0).repeat(n_batches, n_points, 1).to(device)
    w_per_batch_per_d = torch.from_numpy(w).unsqueeze(0).unsqueeze(-1).repeat(n_batches, 1, 3*D).to(device)
    batch_points = points.unsqueeze(-2).repeat(1, 1, n_gaussians, 1).to(device)

    # Define multivariate noraml distributions
    # Compute probability per point
    p_per_point = (1.0 / (np.power(2.0 * np.pi, D / 2.0) * torch.pow(batch_sig[:, :, :, 0], D))) * torch.exp(
        -0.5 * torch.sum(torch.pow((batch_points - batch_mu) / batch_sig, 2.0), axis=3))

    w_p = p_per_point
    Q = w_p  # enforcing the assumption that the sum is 1
    Q_per_d = Q.unsqueeze(-1).repeat(1, 1, 1, D)

    d_pi_all = ((Q - batch_w) / (torch.sqrt(batch_w))).unsqueeze(-1)
    d_pi = torch.cat([torch.max(d_pi_all, axis=1)[0], torch.sum(d_pi_all, axis=1)], axis=2)

    d_mu_all = Q_per_d * (batch_points - batch_mu) / batch_sig
    d_mu = (1.0 / (torch.sqrt(w_per_batch_per_d))) * torch.cat([torch.max(d_mu_all, axis=1)[0],
                                                                torch.min(d_mu_all, axis=1)[0],
                                                                torch.sum(d_mu_all, axis=1)], axis=2)

    d_sig_all = Q_per_d * (torch.pow((batch_points - batch_mu) / batch_sig, 2.0) - 1.0)
    d_sigma = (1.0 / (torch.sqrt(2 * w_per_batch_per_d))) * torch.cat([torch.max(d_sig_all, axis=1)[0],
                                                                       torch.min(d_sig_all, axis=1)[0],
                                                                       torch.sum(d_sig_all, axis=1)], axis=2)

    # number of points  normaliation
    d_pi = d_pi / n_points
    d_mu = d_mu / n_points
    d_sigma = d_sigma / n_points

    if normalize:
        # Power normaliation
        alpha = 0.5
        d_pi = torch.sign(d_pi) * torch.pow(torch.abs(d_pi), alpha)
        d_mu = torch.sign(d_mu) * torch.pow(torch.abs(d_mu), alpha)
        d_sigma = torch.sign(d_sigma) * torch.pow(torch.abs(d_sigma), alpha)

        # L2 normaliation

        d_pi = F.normalize(d_pi, p=2, dim=1)
        d_mu = F.normalize(d_mu, p=2, dim=1)
        d_sigma = F.normalize(d_sigma, p=2, dim=1)

    fv = torch.cat([d_pi, d_mu, d_sigma], axis=2)
    fv = fv.permute(0, 2, 1) #np.transpose(fv, axes=[0, 2, 1])

    return fv


def get_3d_grid_gmm(subdivisions=[5,5,5], variance=0.04):
    """
    Compute the weight, mean and covariance of a gmm placed on a 3D grid
    Input:
        subdivisions: 2 element list of number of subdivisions of the 3D space in each axes to form the grid
        variance: scalar for spherical gmm.p
    Return:
         gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    # n_gaussians = reduce(lambda x, y: x*y,subdivisions)
    n_gaussians = np.prod(np.array(subdivisions))
    step = [ 1.0/(subdivisions[0]),  1.0/(subdivisions[1]),  1.0/(subdivisions[2])]

    means = np.mgrid[ step[0]-1 : 1.0-step[0]: complex(0,subdivisions[0]),
                      step[1]-1 : 1.0-step[1]: complex(0,subdivisions[1]),
                      step[2]-1 : 1.0-step[2]: complex(0,subdivisions[2])]
    means = np.reshape(means, [3,-1]).T
    covariances = variance*np.ones_like(means)
    weights = (1.0/n_gaussians)*np.ones(n_gaussians)
    gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
    gmm.weights_ = np.float32(weights)
    gmm.covariances_ =  np.float32(covariances)
    gmm.means_ =  np.float32(means)
    from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, 'diag')
    return gmm


if __name__ == '__main__':

  batch_size, K, n_points = 64, 8, 128

  # Create random Tensors to hold inputs and outputs
  points = torch.randn(batch_size, n_points, 3, dtype=torch.float32)
  y = torch.randn(batch_size, 1)

  # Construct our model by instantiating the class defined above.
  model = ThreeDmFVNet(K, 40, normalize=True)

  loss_fn = torch.nn.MSELoss(reduction='sum')
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
  for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(points)

    # Compute and print loss
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()