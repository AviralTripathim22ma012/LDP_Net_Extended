import numpy as np
from io_utils_1 import parse_args_test
import test_1_dataset
import ResNet10_1
import torch
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
import random
import warnings
warnings.filterwarnings("ignore", category=Warning)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def test(novel_loader, model, params):
    iter_num = len(novel_loader)
    acc_all_LR = []
    with torch.no_grad():
        for i, (x, _) in enumerate(novel_loader):
            x_query = x[:, params.n_support:, :, :, :].contiguous().view(params.n_way * params.n_query,
                                                                          *x.size()[2:]).cuda()
            x_support = x[:, :params.n_support, :, :, :].contiguous().view(params.n_way * params.n_support,
                                                                            *x.size()[2:]).cuda()
            out_support = model(x_support)
            out_query = model(x_query)

            beta = 0.5
            out_support = torch.pow(out_support, beta)
            out_query = torch.pow(out_query, beta)

            _, c = out_support.size()

            out_support_LR_with_GC = out_support.cpu().numpy()
            out_query_LR_with_GC = out_query.cpu().numpy()
            y = np.tile(range(params.n_way), params.n_support)
            y.sort()

            # Logistic Regression based on complete image
            classifier = LogisticRegression(max_iter=1000).fit(X=out_support_LR_with_GC, y=y)
            pred = classifier.predict(out_query_LR_with_GC)

            pred_a1 = classifier.predict_proba(out_query_LR_with_GC)

            gt = np.tile(range(params.n_way), params.n_query)
            gt.sort()
            acc_LG = np.mean(pred == gt) * 100.
            acc_all_LR.append(acc_LG)

            # Logistic Regression based on random crops



            # crop_classifiers = []
            # for j in range(4):
            #     crop_x_support = torch.nn.functional.interpolate(x_support, scale_factor=0.8, mode='bilinear',
            #                                                       align_corners=False)
            #     crop_x_support = crop_x_support[:, :, j:j + 224, j:j + 224]


            #     crop_x_support = crop_x_support.contiguous().cuda()



            #     crop_out_support = model(crop_x_support)
            #     crop_out_support = torch.pow(crop_out_support, beta)
            #     crop_out_support_LR_with_GC = crop_out_support.cpu().numpy()
            #     crop_classifier = LogisticRegression(max_iter=1000).fit(X=crop_out_support_LR_with_GC, y=y)
            #     crop_classifiers.append(crop_classifier)













    #         crop_classifiers = []
    #         import torchvision.transforms as transforms
    #         crop_transform = transforms.RandomCrop(size=179)
    #         all_predictions = []

    #         for j in range(4):
    #             crop_x_support = torch.nn.functional.interpolate(x_support, scale_factor=0.8, mode='bilinear',
    #                                                               align_corners=False)
    #             # Apply random crop to crop_x_support
    #             cropped_images = []
    #             for image in crop_x_support:
    #                 cropped_image = crop_transform(image)
    #                 cropped_images.append(cropped_image)
    #             crop_x_support = torch.stack(cropped_images)

    #             crop_x_support = crop_x_support.contiguous().cuda()

    #             crop_out_support = model(crop_x_support)
    #             crop_out_support = torch.pow(crop_out_support, beta)
    #             crop_out_support_LR_with_GC = crop_out_support.cpu().numpy()
    #             crop_classifier = LogisticRegression(max_iter=1000).fit(X=crop_out_support_LR_with_GC, y=y)
    #             crop_classifiers.append(crop_classifier)

    #             # Predict probabilities using predict_proba method
    #             crop_pred_proba = crop_classifier.predict_proba(out_query_LR_with_GC)
    #             all_predictions.append(crop_pred_proba)

    #         # Add the predictions from the classifier based on the complete image
    #         complete_image_classifier = LogisticRegression(max_iter=1000).fit(X=out_support_LR_with_GC, y=y)
    #         complete_image_pred_proba = complete_image_classifier.predict_proba(out_query_LR_with_GC)
    #         all_predictions.append(complete_image_pred_proba)

    #         # Calculate average prediction probabilities
    #         avg_pred_proba = np.mean(all_predictions, axis=0)

    #         # Obtain the predicted labels by selecting the class with the highest probability
    #         pred = np.argmax(avg_pred_proba, axis=1)

    # # Calculate accuracy
    # gt = np.tile(range(params.n_way), params.n_query)
    # gt.sort()
    # acc_avg = np.mean(pred == gt) * 100.
    # acc_all_LR.append(acc_avg)
    # print('Average Accuracy: %4.2f%%' % acc_avg)











            crop_classifiers = []
            crop_size = (179, 179)
            for j in range(4):
                crop_x_support = []
                crop_x_query = []

                for image_support, image_query in zip(x_support, x_query):
                    _, h, w = image_support.size()

                    top = random.randint(0, h - crop_size[0])
                    left = random.randint(0, w - crop_size[1])

                    crop_support = image_support[:, top:top+crop_size[0], left:left+crop_size[1]]
                    crop_query = image_query[:, top:top+crop_size[0], left:left+crop_size[1]]

                    crop_x_support.append(crop_support)
                    crop_x_query.append(crop_query)

                crop_x_support = torch.stack(crop_x_support).contiguous().cuda()
                crop_x_query = torch.stack(crop_x_query).contiguous().cuda()

                crop_out_support = model(crop_x_support)
                crop_out_support = torch.pow(crop_out_support, beta)
                crop_out_support_LR_with_GC = crop_out_support.cpu().numpy()
                crop_classifier = LogisticRegression(max_iter=1000).fit(X=crop_out_support_LR_with_GC, y=y)
                crop_classifiers.append(crop_classifier)

            # Combine predictions from all classifiers
            avg_pred = np.zeros_like(pred)
            for k in range(params.n_way * params.n_query):
                crop_preds = []
                for crop_classifier in crop_classifiers:
                    crop_pred = crop_classifier.predict(out_query_LR_with_GC[k].reshape(1, -1))
                    crop_preds.append(crop_pred[0])
                avg_pred[k] = int(stats.mode(crop_preds)[0])

            acc_avg = np.mean(avg_pred == gt) * 100.
            acc_all_LR.append(acc_avg)

    acc_all = np.asarray(acc_all_LR)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('acc: %4.2f%% +- %4.2f%%' % (acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))





                

            # Combine predictions from all classifiers
    #         avg_pred = np.zeros_like(pred)
    #         for k in range(params.n_way * params.n_query):
    #             crop_preds = []
    #             for crop_classifier in crop_classifiers:
    #                 crop_pred = crop_classifier.predict_proba(out_query_LR_with_GC[k].reshape(1, -1))
    #                 crop_preds.append(crop_pred[0])
    #             avg_pred[k] = int(stats.mode(crop_preds)[0])

    #         acc_avg = np.mean(avg_pred == gt) * 100.
    #         acc_all_LR.append(acc_avg)

    # acc_all = np.asarray(acc_all_LR)
    # acc_mean = np.mean(acc_all)
    # acc_std = np.std(acc_all)
    # print('acc: %4.2f%% +- %4.2f%%' % (acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))


if __name__ == '__main__':
    params = parse_args_test()
    setup_seed(params.seed)

    datamgr = test_1_dataset.Eposide_DataManager(data_path=params.current_data_path, num_class=params.current_class,
                                               image_size=params.image_size, n_way=params.n_way,
                                               n_support=params.n_support, n_query=params.n_query,
                                               n_eposide=params.test_n_eposide)
    novel_loader = datamgr.get_data_loader(aug=False)
    model = ResNet10_1.ResNet(list_of_out_dims=params.list_of_out_dims, list_of_stride=params.list_of_stride,
                            list_of_dilated_rate=params.list_of_dilated_rate)

    # test for pretraining model
    tmp = torch.load(params.pretrain_model_path)
    state = tmp['state']
    model.load_state_dict(state)
    model.cuda()
    model.eval()
    test(novel_loader, model, params)

    # test for our method
    tmp = torch.load(params.model_path)
    state_model = tmp['state_model']
    model.load_state_dict(state_model)
    model.cuda()
    model.eval()
    test(novel_loader, model, params)
