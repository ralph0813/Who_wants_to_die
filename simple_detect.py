from models import *
from utils.utils import *

basepath = os.path.dirname(__file__)


def formate_image(img, new_shape=416, color=(128, 128, 128)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
    dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img, ratiow, ratioh, dw, dh


def init_network(cfg, data, weights, img_size=416):
    # Initialize
    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = True  # set False for reproducible results
    # Initialize model
    model = Darknet(cfg, img_size)
    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)
    # Eval mode
    model.to(device).eval()
    # Get classes and colors
    classes = load_classes(parse_data_cfg(data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    init = (classes, colors, model, img_size, device)
    return init


def get_bbox(init, image, out_img_name='out.jpg', img_out_dir='output/', conf_thres=0.5, nms_thres=0.5,
             save_images=True):
    classes, colors, model, img_size, device = init
    img0 = cv2.imread(image)
    img, *_ = formate_image(img0, new_shape=img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
    img /= 255.0

    res = []
    # Get detections
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    pred, _ = model(img)
    det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]
    if det is not None and len(det) > 0:
        # Rescale boxes from 416 to true image size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for *xyxy, conf, cls_conf, cls in det:
            # Add bbox to the image

            label = '%s %.2f' % (classes[int(cls)], conf)
            left, top, right, bottom = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            classname = label.split(" ")[0]
            result = [classname, left, top, right, bottom]
            res.append(result)

            if save_images:
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)])

        if save_images:
            if not os.path.exists(img_out_dir):
                os.mkdir(img_out_dir)
            cv2.imwrite(os.path.join(img_out_dir, out_img_name), img0)

    return res


def load_model():
    out_path = os.path.join('output')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    cfg =  os.path.join(basepath,'cfg/hat_608.cfg')
    data =  os.path.join(basepath,'data/hat_608.data')
    weights =  os.path.join(basepath,'weights/hat_608.weights')
    with torch.no_grad():
        # init
        model = init_network(cfg, data, weights)
    return model


def process_img(ori_img_path, out_img_name='out.jpg', img_out_dir='output/'):
    model = load_model()
    res = get_bbox(model, ori_img_path, out_img_name, img_out_dir)
    return res


if __name__ == '__main__':
    print(process_img('/Users/ralph/github/yolov3/image/May.jpg'))
