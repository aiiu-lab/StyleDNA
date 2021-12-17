import torch
import os
import argparse

from torchvision.utils import save_image
from model.model import pSp, condi
from model.DNAnet import DNAnet
from utils.utils import align_face, totensor
import torch.nn as nn

def main(args):
	device = torch.device(args.device)

	# Load model weights
	DNA = DNAnet().to(device)
	DNA.load_state_dict(torch.load(args.dna), strict=True)
	DNA.eval()

	net = nn.DataParallel(pSp(3, args.enc, args.gan)).to(device)
	net.eval()

	mapper = nn.DataParallel(condi()).to(device)
	mapper.load_state_dict(torch.load(args.map), strict=True)
	mapper.eval()

	# Inference
	mImg = align_face(args.mom_path).convert('RGB')
	dImg = align_face(args.dad_path).convert('RGB')

	testAge =  torch.ones((1,1)) * (args.target_age) /100 
	if args.target_gender == 'male':
		testGen = torch.ones(1, 1).to(device)
	else:
		testGen = torch.zeros(1, 1).to(device)

	with torch.no_grad():
		#print(mImg)
		#print(dImg)
		mImg = totensor(mImg).unsqueeze(0).to(device)
		dImg = totensor(dImg).unsqueeze(0).to(device)

		sW_hat = DNA(net.module.encoder(mImg), net.module.encoder(dImg))
		sW_hat_expand = sW_hat.repeat(18, 1, 1).permute(1, 0, 2)
		sW_hat_delta = mapper(sW_hat_expand, testAge,  testGen)
		sImg_hat = net(sW_hat_expand + sW_hat_delta)

		save_image((sImg_hat+1)/2, f'{args.outputs}/result.png', nrow = 1)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with StyleDNA')

    # Image Path and Saving Path
    parser.add_argument('-m', '--mom_path', required=True, help='Path of mom image')
    parser.add_argument('-d', '--dad_path', required=True, help='Path of dad image')
    parser.add_argument('-a', '--target_gender', help='Predicted child age', default= 'male')
    parser.add_argument('-g', '--target_age', help='Predicted child gender', default = 15)
    parser.add_argument('-o', '--outputs', help='Output directory', default='./result')

    # Model weights Path
    parser.add_argument('-E', '--enc', help='Path of pretrained encoder model', default='./pretrained_model/enc_4_2.pth')
    parser.add_argument('-M', '--map', help='Path of pretrained mapper model', default='./pretrained_model/condi_4_2.pth')
    parser.add_argument('-G', '--gan', help='Path of pretrained stylegan2 model', default='./pretrained_model/stylegan2-ffhq-config-f.pt')
    parser.add_argument('-D', '--dna', help='Path of pretrained DNA model', default='./pretrained_model/10.pth')

    # Device
    parser.add_argument('--device', help='Device to be used by the model (default=cuda:0)', default="cuda")
    args = parser.parse_args()

    ### RUN
    main(args)



