import crypten
import crypten.mpc as mpc
import torch
import nets

@mpc.run_multiprocess(world_size=2)
def run():
    dummy_model = nets.Net2()
    plaintext_model = crypten.load('models/CNN.pth', dummy_model=dummy_model, src=0, map_location=torch.device('cpu'))
    dummy_input = torch.empty((1, 768))
    private_model = crypten.nn.from_pytorch(plaintext_model, dummy_input)
    print('done')

run()
