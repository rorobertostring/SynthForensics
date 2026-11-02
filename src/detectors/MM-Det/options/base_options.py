import argparse


class BaseOption:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-d', '--data-root', type=str, default='./data/DVF_recons', help='the dataset root')
        self.parser.add_argument('--ckpt', type=str, default='./output/weights/model.pth', help='the checkpoint path')
        self.parser.add_argument('--lmm-ckpt', type=str, default='sparklexfantasy/llava-1.5-7b-rfrd', help='the checkpoint of lmm')
        self.parser.add_argument('--lmm-base', type=str, default=None, help='the base model of lmm')
        self.parser.add_argument('--st-ckpt', type=str, default='weights/ViT/vit_base_r50_s16_224.orig_in21k/jx_vit_base_resnet50_224_in21k-6f7c7740.pth', help='the checkpoint of the pretrained checkpoint of hybrid vit in ST branch')
        self.parser.add_argument('--st-pretrained', type=bool, default=True, help='whether to use the pretrained checkpoint of hybrid vit in ST branch')
        self.parser.add_argument('--model-name', type=str, default='MMDet', help='the model name')
        self.parser.add_argument('--expt', type=str, default='MMDet_01', help='the experiment name')
        self.parser.add_argument('--window-size', type=int, default=10, help='window size for video clips')
        self.parser.add_argument('--conv-mode', type=str, default='llava_v1', help='the conversation mode of lmm')
        self.parser.add_argument('--new-tokens', type=int, default=64, help='the number of extracted tokens of the output layer of lmm')
        self.parser.add_argument('--selected-layers', type=int, nargs='+', default=[-1], help='the selected layers for feature of lmm')
        self.parser.add_argument('--interval', type=int, default=200, help='the interval between cached mm representataions of lmm, only available for caching')
        self.parser.add_argument('--load-8bit', action='store_true', help='whether load lmm of 8 bit')
        self.parser.add_argument('--load-4bit', action='store_true', help='whether load lmm of 4 bit')
        self.parser.add_argument('--seed', type=int, default=1, help='random seed')
        self.parser.add_argument('--gpus', type=int, default=1, help='number for gpus')
        self.parser.add_argument('--cache-mm', action='store_true', help='whether load mm encoder or use cached representations')
        self.parser.add_argument('--mm-root', type=str, default='./data/DVF_mm_representations', help='the path of cached mm representations')
        self.parser.add_argument('--debug', action='store_true', help='debug mode')

        
    def parse(self):
        return self.parser.parse_args()