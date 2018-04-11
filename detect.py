from __future__ import division
import cv2
import numpy as np
import pickle
from keras import backend as K
from keras_frcnn import resnet as nn
from keras.layers import Input, TimeDistributed, Dense, Dropout, Activation
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn.MyLayer import MyLayer
from numpy.linalg import norm
# import shutil


with open('Model/config.pickle', 'rb') as f_in:
    C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))
    return (real_x1, real_y1, real_x2, real_y2)



class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

word = np.loadtxt('ImageNet2017/word_w2v.txt', dtype='float32', delimiter=',')
wordname_lines = open('ImageNet2017/cls_names.txt').read().split("\n")
word_alphabetical_index = {}
v = 1
unseenind = len(class_mapping)
class_mapping_unseen = {}
for idx in range(int(len(wordname_lines)) - 1):
    word_alphabetical_index[wordname_lines[idx]] = v
    v = v + 1
    if wordname_lines[idx] not in class_mapping:
        class_mapping_unseen[wordname_lines[idx]] = unseenind
        unseenind = unseenind + 1

# Getting only seen word vector
word_seen = np.zeros((word.shape[0], len(class_mapping)))
word_all = np.zeros((word.shape[0], word.shape[1]))
for key in (class_mapping.keys()):
    pos = word_alphabetical_index[key] - 1
    word_seen[:, class_mapping[key]] = word[:, pos]
    word_all[:, class_mapping[key]] = word[:, pos]

for key in (class_mapping_unseen.keys()):
    pos = word_alphabetical_index[key] - 1
    word_all[:, class_mapping_unseen[key]] = word[:, pos]


word_all_ex = np.expand_dims(norm(word_all, axis=0), 0)
class_mapping_ = class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
class_mapping_unseen_ = class_mapping_unseen
class_mapping_unseen = {v: k for k, v in class_mapping_unseen.items()}

class_to_color = {class_mapping[v]: np.array([255, 0, 255]) for v in class_mapping}
class_to_color_u = {class_mapping_unseen[v]: np.array([0, 255, 1]) for v in class_mapping_unseen}
class_to_color.update(class_to_color_u) 
#class_to_color = dict(class_to_color.items() + class_to_color_u.items())

C.num_rois = 32

input_shape_img = (None, None, 3)



img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))

# define the base network (resnet here)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier = Model([img_input, roi_input], classifier)

model_classifier.layers.pop()
model_classifier.layers.pop()
nb_classes = len(class_mapping)

resnetlast = (model_classifier.layers[-1].output)

out_class = TimeDistributed(Dense(int(resnetlast.shape[2]), kernel_initializer='uniform'))(resnetlast)
out_class = Activation('relu')(out_class)
out_class = Dropout(.5)(out_class)
out_class = TimeDistributed(Dense(word.shape[0], activation='linear', kernel_initializer='uniform'))(out_class)
out_class = TimeDistributed(Dense(word.shape[0], activation='linear', kernel_initializer='uniform'),
                            name='dense_class_{}'.format(nb_classes))(out_class)
out_class = MyLayer(output_dim=word_all.shape[1])(out_class)
out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
                           name='dense_regress_{}'.format(nb_classes))(resnetlast)

model_classifier = Model([img_input, roi_input], [out_class, out_regr])


print('Loading trained weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

    
model_classifier_Mylayer = Model(inputs=model_classifier.input,
                                 outputs=model_classifier.layers[-3].output)

all_imgs = []

classes = {}

# outputfp = open('prediction.txt', 'w')

lines = open('sample_input.txt').read().split("\n")

bbox_threshold = .2
bbox_threshold_unseen = bbox_threshold

visualise = False  # True
for idx in range(int(len(lines))-1):
    aline = lines[idx].split(" ")
    im_id = aline[1]
    filepath = aline[0] + '.JPEG'

    print('{}/{}'.format(im_id, len(lines) - 1))

    img = cv2.imread(filepath)

    X, ratio = format_img(img, C)
    X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)

    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7) # v.7

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}

    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // C.num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier.predict([X, ROIs]) #F

        # to calculate cosine similarity
        model_classifier_Mylayer_output = model_classifier_Mylayer.predict([X, ROIs]) #F
        b = np.repeat(word_all_ex, 32, 0)
        a = np.transpose(norm(model_classifier_Mylayer_output, axis=2))
        P_cls[0,:,:] = P_cls[0,:,:] / np.multiply(a, b)

        for ii in range(P_cls.shape[1]):

            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (word_seen.shape[1] - 1):
                continue

            maxind = np.argmax(P_cls[0, ii, :])
            seenmaxind = np.argmax(P_cls[0, ii, :(word_seen.shape[1])])
            unseenmaxind = word_seen.shape[1] + np.argmax(P_cls[0, ii, word_seen.shape[1]:])

            cls_name = class_mapping[seenmaxind]
            cls_name_unseen = class_mapping_unseen[unseenmaxind]


            if P_cls[0, ii, unseenmaxind] > bbox_threshold_unseen:
                final_cls_name = cls_name_unseen
                final_cls_ind = unseenmaxind
                if final_cls_name not in bboxes:
                    bboxes[final_cls_name] = []
                    probs[final_cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = seenmaxind
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[final_cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[final_cls_name].append(P_cls[0, ii, final_cls_ind])

            # Only seen classification
            if P_cls[0, ii, seenmaxind] > bbox_threshold:
                final_cls_name = cls_name
                final_cls_ind = seenmaxind

                if final_cls_name not in bboxes:
                    bboxes[final_cls_name] = []
                    probs[final_cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = seenmaxind
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[final_cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[final_cls_name].append(P_cls[0, ii, final_cls_ind])


    all_dets = []
    all_dets_box = []

    thold = []
    index = 0
    for key in bboxes:

        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)

        for jk in range(new_boxes.shape[0]):
            thold.append(new_probs[jk])
            index = index + 1
            (x1, y1, x2, y2) = new_boxes[jk, :]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

            all_dets.append((key, 100 * new_probs[jk]))
            all_dets_box.append(new_boxes[jk])


            if key in class_mapping_unseen_.keys():
                # outputfp.write(str(im_id) + ' ' + str(word_alphabetical_index[key]) + ' ' + str(new_probs[jk]) + ' ' + str(
                #     real_x1) + ' ' + str(real_y1) + ' ' + str(real_x2) + ' ' + str(real_y2) + '\n')

                cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                              (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])), 2)

                textLabel = '{}: {}'.format(key, int(100 * new_probs[jk]))

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (real_x1, real_y1 - 0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, .7, (0, 0, 0), 1)


    # for key in sorted(all_dets, key=lambda tup: (-tup[1], tup[0])):
    #     if key[0] in class_mapping_unseen_.keys():
    #         print('unseen:{}'.format(key))
    #     else:
    #         print('seen  :{}'.format(key))

    cv2.imwrite('Dataset/Sampleoutput/'+im_id+'.jpg',img)

    if visualise:
        cv2.imshow('img', img)
        cv2.waitKey(0)

# outputfp.close()