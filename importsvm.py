import crypten
import crypten.mpc as mpc
import torch
import nets
import torch.functional as F
import crypten.communicator as comm

@mpc.run_multiprocess(world_size=2)
def run():
    dummy_model = nets.Net6()
    #plaintext_model = crypten.load('models/CNN.pth', dummy_model=dummy_model, src=0, map_location=torch.device('cpu'))
    plaintext_model = crypten.load('checkpoint.pth', dummy_model=dummy_model, src=0)
    dummy_input = torch.empty((1, 1, 768))
    dummy_input.to('cuda')
    private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
    private_model.encrypt()
    private_model.eval()
    input = torch.rand((1, 1, 768))
    input = crypten.cryptensor(input, src=0)
    classification = private_model(input)
    print(classification)
    print('done')

@mpc.run_multiprocess(world_size=2)
def test_mp2d():
    dummy_model = nets.Net5()
    x_small = torch.rand(100, 1, 28, 28)
    y_small = torch.randint(1, (100,))
    label_eye = torch.eye(2)
    y_one_hot = label_eye[y_small]
    x_train = crypten.cryptensor(x_small, src=0)
    y_train = crypten.cryptensor(y_one_hot)
    #plaintext_model = crypten.load('models/CNN.pth', dummy_model=dummy_model, src=0, map_location=torch.device('cpu'))
    dummy_input = torch.empty((1, 1, 28, 28))
    private_model = crypten.nn.from_pytorch(dummy_model, dummy_input)
    private_model.encrypt()
    private_model.train()
    loss = crypten.nn.MSELoss()

    lr = 0.001
    num_epochs = 2
    for i in range(num_epochs):
        output = private_model(x_train)
        loss_value= loss(output, y_train)
        private_model.zero_grad()
        loss_value.backward()
        private_model.update_parameters(lr)
        print("Epoch: {0:d} Loss: {1:.4f}".format(i, loss_value.get_plain_text()))
    print('done')

@mpc.run_multiprocess(world_size=2)
def test_mp1d():
    x_small = torch.rand(10, 3, 28)
    mp = torch.nn.MaxPool1d(2, return_indices=True)
    res, ind = mp(x_small)
    print(ind)
    x_crypt = mpc.MPCTensor(x_small, ptype=mpc.arithmetic)

    

crypten.init()
# test_mp1d()
#test_mp2d()
run()
