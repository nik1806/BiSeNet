import cv2
import argparse
from model.build_BiSeNet import BiSeNet
import os
import torch
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import reverse_one_hot, get_label_info, colour_code_segmentation

def predict_on_image(model, args, image):
    '''
        run inference and return the resultant image
    '''
    # pre-processing on image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resize = iaa.Scale({'height': args.crop_height, 'width': args.crop_width})
    resize_det = resize.to_deterministic()
    image = resize_det.augment_image(image)
    image = Image.fromarray(image).convert('RGB')
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image).unsqueeze(0)
    # read csv label path
    label_info = get_label_info(args.csv_path)
    # predict
    model.eval()
    predict = model(image).squeeze()
    predict = reverse_one_hot(predict)
    # predict = colour_code_segmentation(np.array(predict), label_info)
    predict = colour_code_segmentation(np.array(predict.cpu()), label_info)
    predict = cv2.resize(np.uint8(predict), (960, 720))
    # cv2.imwrite(args.save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
    return predict

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', action='store_true', default=False, help='predict on image')
    parser.add_argument('--video', action='store_true', default=False, help='predict on video')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='The path to the pretrained weights of model')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--num_classes', type=int, default=12, help='num of object classes (with void)')
    parser.add_argument('--data', default=None, help='Path to image or video for prediction')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=960, help='Width of cropped/resized input image to network')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use gpu for training')
    parser.add_argument('--csv_path', type=str, default=None, required=True, help='Path to label info csv file')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Path to save predict image')

    args = parser.parse_args(params)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    # predict on image
    if args.image:
        # read image
        image = cv2.imread(str(args.data), -1)

        # run model
        res_image = predict_on_image(model, args, image)
        
        cv2.imwrite(args.save_path, res_image, cv2.COLOR_RGB2BGR)
        # display the result
        cv2.imshow("BiSeNet window", res_image)
        cv2.waitKey(0)        


    # predict on video
    if args.video:

        cap = cv2.VideoCapture(args.data)
        
        # while video source is available, loop over the frames
        while cap.isOpened():
        
            ret, frame = cap.read()
            
            # run model
            res_frame = predict_on_image(model, args, frame)

            # display frames
            cv2.imshow("img", res_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    params = [
        '--video',
        # '--image',
        '--data', 0,
        '--checkpoint_path', '/path/to/ckpt',
        '--cuda', '0',
        '--csv_path', '/data/sqy/CamVid/class_dict.csv',
        '--save_path', 'demo.png',
        '--context_path', 'resnet18'
    ]

    main(params)
   