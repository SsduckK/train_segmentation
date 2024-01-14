import dataset_util as du
import seaborn as sns
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import matplotlib.pyplot as plt
import numpy as np

colors = sns.color_palette(None, len(du.class_names))

def fuse_with_pil(images):
    widths = (image.shape[1] for image in images)
    heights = (image.shape[0] for image in images)
    total_width = sum(widths)
    max_height = max(heights)

    new_im = PIL.Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        pil_image = PIL.Image.fromarray(np.uint8(im))
        new_im.paste(pil_image, (x_offset, 0))
        x_offset += im.shape[1]
    return new_im


def give_color_to_annotation(annotation):
    seg_img = np.zeros( (annotation.shape[0], annotation.shape[1], 3)).astype('float')

    for c in range(12):
        segc = (annotation == c)
        seg_img[:, :, 0] += segc*( colors[c][0] * 255.0)
        seg_img[:, :, 1] += segc*( colors[c][1] * 255.0)
        seg_img[:, :, 2] += segc*( colors[c][2] * 255.0)

    return seg_img


def show_predictions(image, labelmaps, titles, iou_list, dice_score_list):
    true_img = give_color_to_annotation(labelmaps[1])
    pred_img = give_color_to_annotation(labelmaps[0])

    image = image + 1
    image = image * 127.5
    images = np.uint8([image, pred_img, true_img])

    metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list)) if iou > 0.0]
    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)

    display_string_list = ["{}:IOU: {} Dice Score: {}".format(du.class_names[idx], iou, dice_score) for idx, iou, dice_score in metrics_by_id]
    display_string = "\n\n".join(display_string_list)

    plt.figure(figsize=(15, 4))

    for idx, im in enumerate(images):
        plt.subplot(1, 3, idx+1)
        if idx == 1:
            plt.xlabel(display_string)
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[idx], fontsize=12)
        plt.imshow(im)


def show_annotation_and_image(image, annotation):
    new_ann = np.argmax(annotation, axis=2)
    seg_img = give_color_to_annotation(new_ann)

    image = image + 1
    image = image * 127.5
    image = np.uint8(image)
    images = [image, seg_img]
    fused_img = fuse_with_pil(images)
    plt.imshow(fused_img)


def list_show_annotation(dataset):
    ds = dataset.unbatch()
    ds = ds.shuffle(buffer_size=100)

    plt.figure(figsize=(25, 15))
    plt.title("Image And Annotation")
    plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)

    for idx, (image, annotation) in enumerate(ds.take(9)):
        plt.subplot(3, 3, idx + 1)
        plt.yticks([])
        plt.xticks([])
        show_annotation_and_image(image.numpy(), annotation.numpy())
    plt.show()